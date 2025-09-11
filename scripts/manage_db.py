#!/usr/bin/env python3
"""
Database management script for VectorBeat.
Provides utilities for database operations like reset, stats, and maintenance.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import asyncio

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
from database.connection import get_database
from database.schema import reset_database, get_database_stats, create_tables

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def reset_db():
    """Reset the database by dropping and recreating all tables."""
    try:
        logger.info("Resetting database...")
        db = get_database()
        await reset_database(db)
        await create_tables(db)
        logger.info("Database reset completed successfully")
    except Exception as e:
        logger.error(f"Failed to reset database: {str(e)}")
        raise


async def show_stats():
    """Show database statistics."""
    try:
        logger.info("Fetching database statistics...")
        db = get_database()
        stats = await get_database_stats(db)
        
        print("\n" + "="*50)
        print("VECTORBEAT DATABASE STATISTICS")
        print("="*50)
        print(f"Total Tracks: {stats['total_tracks']:,}")
        print(f"Unique Artists: {stats['total_artists']:,}")
        print(f"Unique Genres: {stats['total_genres']:,}")
        
        if stats['year_range']:
            print(f"Year Range: {stats['year_range']['min']} - {stats['year_range']['max']}")
        
        if stats['last_updated']:
            print(f"Last Updated: {stats['last_updated']}")
        
        if stats['database_size_mb']:
            print(f"Database Size: {stats['database_size_mb']:.2f} MB")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {str(e)}")
        raise


async def optimize_db():
    """Optimize database performance."""
    try:
        logger.info("Optimizing database...")
        db = get_database()
        
        # This is a placeholder - LanceDB doesn't have explicit optimization commands
        # but we can check table health and suggest maintenance
        table_names = db.table_names()
        logger.info(f"Found {len(table_names)} tables: {table_names}")
        
        logger.info("Database optimization completed")
        
    except Exception as e:
        logger.error(f"Failed to optimize database: {str(e)}")
        raise


async def backup_db():
    """Create a backup of the database."""
    try:
        import shutil
        from datetime import datetime
        
        db_path = os.getenv("LANCE_DB_PATH", "./data/lancedb")
        backup_path = f"{db_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Creating database backup to {backup_path}...")
        shutil.copytree(db_path, backup_path)
        logger.info(f"Database backup created successfully at {backup_path}")
        
    except Exception as e:
        logger.error(f"Failed to create database backup: {str(e)}")
        raise


async def restore_db(backup_path: str):
    """Restore database from backup."""
    try:
        import shutil
        
        db_path = os.getenv("LANCE_DB_PATH", "./data/lancedb")
        
        if not os.path.exists(backup_path):
            raise ValueError(f"Backup path does not exist: {backup_path}")
        
        logger.info(f"Restoring database from {backup_path}...")
        
        # Remove existing database
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        # Copy backup to database location
        shutil.copytree(backup_path, db_path)
        logger.info("Database restored successfully")
        
    except Exception as e:
        logger.error(f"Failed to restore database: {str(e)}")
        raise


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="VectorBeat Database Management")
    parser.add_argument("--action", required=True,
                       choices=["reset", "stats", "optimize", "backup", "restore"],
                       help="Action to perform")
    parser.add_argument("--backup-path", help="Path to backup for restore action")
    parser.add_argument("--confirm", action="store_true",
                       help="Confirm destructive operations")
    
    args = parser.parse_args()
    
    try:
        if args.action == "reset":
            if not args.confirm:
                print("WARNING: This will delete all data in the database!")
                print("Use --confirm to proceed with the reset.")
                sys.exit(1)
            await reset_db()
            
        elif args.action == "stats":
            await show_stats()
            
        elif args.action == "optimize":
            await optimize_db()
            
        elif args.action == "backup":
            await backup_db()
            
        elif args.action == "restore":
            if not args.backup_path:
                print("ERROR: --backup-path is required for restore action")
                sys.exit(1)
            if not args.confirm:
                print("WARNING: This will replace the current database!")
                print("Use --confirm to proceed with the restore.")
                sys.exit(1)
            await restore_db(args.backup_path)
        
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
