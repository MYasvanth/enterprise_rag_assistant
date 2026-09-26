"""
Dead Letter Queue (DLQ) Handler
Manages storage and processing of permanently failed ingestion jobs.
Jobs are sent to DLQ when they exceed max retries and cannot be processed automatically.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, delete

from ...db.base import async_get_db
from ...models.dlq_entry import DLQEntry  # We'll need this model, but referencing it for completeness

logger = logging.getLogger(__name__)


class DLQHandler:
    """Handler for managing dead letter queue entries."""

    @staticmethod
    async def add_to_dlq(job_id: str, file_path: str, tenant_id: str, error: str) -> None:
        """
        Add a failed job to the dead letter queue.
        Stores metadata about the failure for later analysis and manual reprocessing.
        """
        async for db in async_get_db():
            try:
                dlq_entry = DLQEntry(
                    job_id=job_id,
                    file_path=file_path,
                    error_message=error,
                    tenant_id=tenant_id,
                    failed_at=datetime.utcnow(),
                    reprocessed=False
                )
                db.add(dlq_entry)
                await db.commit()
                logger.critical(f"Added job {job_id} to DLQ: {file_path}")
            except Exception as e:
                logger.error(f"Failed to add job {job_id} to DLQ: {str(e)}", exc_info=True)

    @staticmethod
    async def list_dlq_entries(
        db: AsyncSession,
        include_reprocessed: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[DLQEntry]:
        """
        List all entries in the DLQ.
        Optionally include entries that have already been reprocessed.
        """
        query = select(DLQEntry).order_by(DLQEntry.failed_at.desc())
        
        if not include_reprocessed:
            query = query.where(DLQEntry.reprocessed == False)
            
        query = query.limit(limit).offset(offset)
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def mark_as_reprocessed(db: AsyncSession, entry_id: int) -> bool:
        """
        Mark a DLQ entry as reprocessed after manual intervention.
        Returns True if the entry was found and updated.
        """
        entry = await db.get(DLQEntry, entry_id)
        if not entry:
            logger.warning(f"DLQ entry {entry_id} not found")
            return False
            
        entry.reprocessed = True
        entry.reprocessed_at = datetime.utcnow()
        await db.commit()
        logger.info(f"Marked DLQ entry {entry_id} (job {entry.job_id}) as reprocessed")
        return True

    @staticmethod
    async def delete_entry(db: AsyncSession, entry_id: int) -> bool:
        """
        Delete a DLQ entry after it's been processed and analyzed.
        """
        entry = await db.get(DLQEntry, entry_id)
        if not entry:
            logger.warning(f"DLQ entry {entry_id} not found for deletion")
            return False
            
        await db.delete(entry)
        await db.commit()
        logger.info(f"Deleted DLQ entry {entry_id}")
        return True

    @staticmethod
    async def get_stats(db: AsyncSession) -> Dict[str, int]:
        """
        Get statistics about DLQ entries: total, unprocessed, reprocessed.
        """
        total = await db.execute(select(DLQEntry))
        total_count = total.scalars().all()
        
        unprocessed = await db.execute(select(DLQEntry).where(DLQEntry.reprocessed == False))
        unprocessed_count = unprocessed.scalars().all()
        
        return {
            "total": len(total_count),
            "unprocessed": len(unprocessed_count),
            "reprocessed": len(total_count) - len(unprocessed_count)
        }

    @staticmethod
    async def retry_dlq_entry(entry_id: int) -> bool:
        """
        Attempt to requeue a DLQ entry for processing.
        This would re-add the original ingestion task to Celery.
        """
        async for db in async_get_db():
            entry = await db.get(DLQEntry, entry_id)
            if not entry:
                logger.warning(f"Cannot retry DLQ entry {entry_id}: not found")
                return False
                
            if entry.reprocessed:
                logger.warning(f"Cannot retry DLQ entry {entry_id}: already reprocessed")
                return False

            try:
                # Import here to avoid circular imports
                from .ingestion_tasks import ingest_document_task
                # Re-queue the ingestion task with the original parameters
                ingest_document_task.delay(
                    entry.job_id,
                    entry.file_path,
                    ""  # tenant_id would need to be stored in DLQ entry too
                )
                logger.info(f"Requeued DLQ entry {entry_id} (job {entry.job_id})")
                return True
            except Exception as e:
                logger.error(f"Failed to requeue DLQ entry {entry_id}: {str(e)}")
                return False