"""
Ingestion Job Service
Handles creation, status updates, and retries of document ingestion jobs.
Works with SQLAlchemy IngestionJob model to track async processing state.
"""

import logging
from uuid import UUID
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ...models.ingestion_job import IngestionJob
from ...db.session import get_db

logger = logging.getLogger(__name__)


class IngestionJobService:
    """Service for managing ingestion job lifecycle."""

    @staticmethod
    async def create_job(
        db: AsyncSession,
        document_id: UUID,
        tenant_id: UUID
    ) -> IngestionJob:
        """
        Create a new ingestion job with REQUESTED status.
        """
        job = IngestionJob(
            document_id=document_id,
            tenant_id=tenant_id,
            status="REQUESTED",
            chunks_processed=0,
            retry_count=0
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        logger.info(f"Created ingestion job {job.id} for document {document_id}")
        return job

    @staticmethod
    async def get_job(db: AsyncSession, job_id: UUID) -> Optional[IngestionJob]:
        """
        Retrieve a job by its ID.
        """
        result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def update_status(
        db: AsyncSession,
        job_id: UUID,
        status: str,
        chunks_processed: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> Optional[IngestionJob]:
        """
        Update job status and related fields.
        Status can be: REQUESTED | VALIDATING | PROCESSING | SUCCEEDED | FAILED
        """
        update_data: Dict[str, Any] = {"status": status, "updated_at": datetime.utcnow()}
        
        if chunks_processed is not None:
            update_data["chunks_processed"] = chunks_processed
        if error_message is not None:
            update_data["error_message"] = error_message

        await db.execute(
            update(IngestionJob)
            .where(IngestionJob.id == job_id)
            .values(**update_data)
        )
        await db.commit()
        
        logger.info(f"Updated job {job_id} to status: {status}")
        return await IngestionJobService.get_job(db, job_id)

    @staticmethod
    async def retry_job(db: AsyncSession, job_id: UUID) -> Optional[IngestionJob]:
        """
        Retry a failed job: increment retry count, reset status to REQUESTED.
        Returns None if job not found or cannot be retried.
        """
        job = await IngestionJobService.get_job(db, job_id)
        if not job:
            return None
            
        if job.status != "FAILED":
            logger.warning(f"Cannot retry job {job_id}: not in FAILED status (current: {job.status})")
            return None
            
        if job.retry_count >= 3:  # Max 3 retries
            logger.warning(f"Cannot retry job {job_id}: max retries exceeded ({job.retry_count}/3)")
            return None

        # Reset for retry
        update_data = {
            "status": "REQUESTED",
            "retry_count": job.retry_count + 1,
            "error_message": None,
            "chunks_processed": 0,
            "updated_at": datetime.utcnow()
        }
        
        await db.execute(
            update(IngestionJob)
            .where(IngestionJob.id == job_id)
            .values(**update_data)
        )
        await db.commit()
        
        logger.info(f"Retrying job {job_id} (attempt {job.retry_count + 1}/3)")
        return await IngestionJobService.get_job(db, job_id)

    @staticmethod
    async def list_jobs(
        db: AsyncSession,
        tenant_id: Optional[UUID] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[IngestionJob]:
        """
        List jobs with optional filtering by tenant and status.
        """
        query = select(IngestionJob)
        
        if tenant_id:
            query = query.where(IngestionJob.tenant_id == tenant_id)
        if status:
            query = query.where(IngestionJob.status == status)
            
        query = query.order_by(IngestionJob.created_at.desc()).limit(limit).offset(offset)
        result = await db.execute(query)
        return list(result.scalars().all())