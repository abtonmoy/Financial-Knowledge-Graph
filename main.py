"""Main entry point for the Financial Knowledge Graph application."""

import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_config
from .engine.processing_engine import FinancialKGEngine
from .api.routes import router, set_engine
from .cli.commands import CLIHandler

# Initialize FastAPI app
app = FastAPI(
    title="Financial Knowledge Graph API",
    description="Open Source Financial Knowledge Graph with RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure later properly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Financial KG"])

# Global engine instance
engine_instance = None

@app.on_event("startup")
async def startup_event():
    """Initialize the engine on startup."""
    global engine_instance
    
    print("Starting Financial Knowledge Graph API...")
    config = get_config()
    
    try:
        # Initialize the engine
        engine_instance = FinancialKGEngine(device=config.DEVICE)
        
        # Set engine in routes
        set_engine(engine_instance)
        
        print("API ready!")
        print(f"API docs available at http://localhost:{config.API_PORT}/docs")
        
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global engine_instance
    
    if engine_instance:
        print("Shutting down Financial Knowledge Graph API...")
        try:
            # Unload models to free memory
            engine_instance.model_manager.unload_all_models()
            print("Models unloaded successfully")
        except Exception as e:
            print(f"Error during shutdown: {e}")

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "Financial Knowledge Graph API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

def main():
    """Command line interface."""
    config = get_config()
    
    if len(sys.argv) < 2:
        print("Financial Knowledge Graph - Modular Version")
        print("\nUsage:")
        print("  python -m financial_kg.main server                    # Start web server")
        print("  python -m financial_kg.main process <file_path>       # Process a document")
        print("  python -m financial_kg.main query '<question>'        # Ask a question")
        print("  python -m financial_kg.main entities [type]           # List entities")
        print("  python -m financial_kg.main audit                     # Run audit checks")
        print("  python -m financial_kg.main stats                     # Show statistics")
        print("  python -m financial_kg.main test                      # Run with sample data")
        print("  python -m financial_kg.main clear                     # Clear all data")
        print("\nConfiguration:")
        print(f"  Database: {config.DATABASE_PATH}")
        print(f"  Vector Store: {config.VECTOR_STORE_PATH}")
        print(f"  Device: {config.DEVICE}")
        return
    
    command = sys.argv[1]
    cli_handler = CLIHandler()
    
    if command == "server":
        print("Starting Financial KG server...")
        print(f"API docs will be available at http://localhost:{config.API_PORT}/docs")
        uvicorn.run(
            "financial_kg.main:app", 
            host=config.API_HOST, 
            port=config.API_PORT,
            reload=False  # Set to True for development
        )
    
    elif command == "process" and len(sys.argv) > 2:
        file_path = sys.argv[2]
        cli_handler.process_document(file_path)
    
    elif command == "query" and len(sys.argv) > 2:
        question = sys.argv[2]
        cli_handler.query_documents(question)
    
    elif command == "entities":
        entity_type = sys.argv[2] if len(sys.argv) > 2 else None
        cli_handler.list_entities(entity_type)
    
    elif command == "audit":
        cli_handler.run_audit()
    
    elif command == "stats":
        cli_handler.show_statistics()
    
    elif command == "test":
        cli_handler.run_test()
    
    elif command == "clear":
        cli_handler.clear_data()
    
    else:
        print(f"Unknown command: {command}")
        print("Run 'python -m financial_kg.main' for usage information.")

if __name__ == "__main__":
    main()