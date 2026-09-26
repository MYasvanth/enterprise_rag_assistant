"""
Celery Ingestion Tasks
Async worker tasks for document ingestion processing.
Handles the full pipeline: validation → parsing → chunking → embedding storage.
"""

import logging
from uuid import UUID
from celery import Celery, shared_task
from celery.utils.log import get_task_logger
from sqlalchemy.ext.asyncio import AsyncSession

from ..ingestion import DocumentIngestionPipeline
from ...embedding.embedding import EmbeddingManager
from .job_service import IngestionJobService
from ...db.session import async_get_db
from ...config.settings import get_settings

logger = get_task_logger(__name__)
settings = get_settings()

# Initialize Celery
celery = Celery('ingestion_tasks')
celery.config_from_object(settings.celery_config)

# Initialize shared pipeline and embedding manager
ingestion_pipeline = DocumentIngestionPipeline()
embedding_manager = EmbeddingManager()


@shared_task(bind=True, max_retries=3, name='ingest_document_task')
def ingest_document_task(
    self,
    job_id: str,
    file_path: str,
    tenant_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> None:
    """
    Celery task to process a document ingestion job asynchronously.
    Full pipeline: VALIDATING → PROCESSING → SUCCEEDED/FAILED.
    
    Args:
        job_id: UUID string of the ingestion job
        file_path: Path to the file to process
        tenant_id: Tenant UUID for isolation
        chunk_size: Chunk size for text splitting
        chunk_overlap: Chunk overlap for text splitting
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        _process_document(
            job_id=UUID(job_id),
            file_path=file_path,
            tenant_id=UUID(tenant_id),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            task=self
        )
    )


async def _process_document(
    job_id: UUID,
    file_path: str,
    tenant_id: UUID,
    chunk_size: int,
    chunk_overlap: int,
    task
) -> None:
    """Async implementation of the document processing pipeline."""
    # Get database session
    async for db in async_get_db():
        try:
            # Update job to VALIDATING
            await IngestionJobService.update_status(db, job_id, "VALIDATING")
            logger.info(f"Job {job_id}: Starting validation of {file_path}")

            # Update chunking parameters
            ingestion_pipeline.set_chunk_params(chunk_size, chunk_overlap)

            # Update to PROCESSING
            await IngestionJobService.update_status(db, job_id, "PROCESSING")
            logger.info(f"Job {job_id}: Starting document processing")

            # Run full ingestion pipeline
            chunks = ingestion_pipeline.run_pipeline(file_path)
            
            # Update chunks processed count
            await IngestionJobService.update_status(
                db, job_id, "PROCESSING", chunks_processed=len(chunks)
            )
            logger.info(f"Job {job_id}: Generated {len(chunks)} chunks, adding to vector store")

            # Add chunks to vector store with tenant isolation
            await embedding_manager.add_documents(chunks, str(tenant_id))

            # Mark job as SUCCEEDED
            await IngestionJobService.update_status(
                db, job_id, "SUCCEEDED", chunks_processed=len(chunks)
            )
            logger.info(f"Job {job_id}: Successfully completed ingestion")

        except Exception as e:
            error_msg = f"Failed to process document: {str(e)}"
            logger.error(f"Job {job_id}: {error_msg}", exc_info=True)
            
            # Update job to FAILED
            await IngestionJobService.update_status(
                db, job_id, "FAILED", error_message=error_msg
            )
            
            # Retry with backoff if we haven't exceeded max retries
            try:
                job = await IngestionJobService.get_job(db, job_id)
                if job and job.retry_count < 3:
                    retry_delay = 60 * (job.retry_count + 1)  # 60s, 120s, 180s
                    raise task.retry(exc=e, countdown=retry_delay)
            except Exception as retry_e:
                logger.error(f"Job {job_id}: Could not retry, max retries exceeded: {str(retry_e)}")
            
            # Send to DLQ if max retries reached
            await send_to_dlq(job_id, file_path, str(tenant_id), str(e))
            return


async def send_to_dlq(job_id: UUID, file_path: str, tenant_id: str, error: str) -> None:
    """Send failed job to dead letter queue for manual processing."""
    logger.critical(f"Job {job_id} sent to DLQ: {file_path}, error: {error}")
    # DLQ implementation would store this in a separate queue/table for monitoring
    from .dlq_handler import DLQHandler
    await DLQHandler.add_to_dlq(str(job_id), file_path, tenant_id, error)


@shared_task(name='retry_failed_jobs')
def retry_failed_jobs() -> None:
    """Periodic task to retry FAILED jobs that haven't exceeded max retries."""
    import asyncio
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_retry_failed_jobs())


async def _retry_failed_jobs() -> None:
    """Async implementation of failed jobs retry scheduler."""
    async for db in async_get_db():
        # Get all FAILED jobs
        failed_jobs = await IngestionJobService.list_jobs(db, status="FAILED")
        logger.info(f"Found {len(failed_jobs)} failed jobs to check for retry")
        
        for job in failed_jobs:
            if job.retry_count < 3:
                logger.info(f"Queueing retry for job {job.id} (retry {job.retry_count + 1}/3)")
                # Re-queue the task - in a real implementation you'd need the original file_path
                # This would typically be stored with the job metadata
                ingest_document_task.delay(
                    str(job.id),
                    "",  # file_path would need to be retrieved from job/document
                    str(job.tenant_id)
                )