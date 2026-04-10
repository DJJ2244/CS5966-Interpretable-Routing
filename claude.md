# Overview
This is a pipeline for a machine learning project. the main entry point will be through cli.py which will act as a CLI which opens up all of the necessary commands. Some but not all of the commands will be used by .sh files which will be used in jobs files used in SLURM.

## Database
- sqlite
- init database creates the sqlite file

## DAOs

DAOs are the only layer that constructs queries or reads raw database results. No business logic, service, or controller should touch a raw row — only domain objects returned by a DAO.

**Structure**
- One DAO per entity. Each owns its table name, field name constants, and all queries for that entity.
- No shared base class or generic interface. Each DAO defines exactly the operations its entity needs.
- Each DAO has a private mapping function that converts a raw row into a domain object. Callers always receive domain objects, never raw rows. Mapping is manual and eager — no ORM, no reflection.

**Query safety**
- All queries must be parameterized. Never interpolate values directly into a query string.
- Reference field names through named constants, not inline string literals.

**Logic**
- DAOs only contain database related logic
- DAOs are the only place where queries are ran on the database
- No SQL (conn.execute, executemany, etc.) outside of daos/ — the exception is DDL and seeding in database_util.py

## Jobs
Inference
Test Model Code
Calculate Route LLM Threshold
Extract Activations
Train SAE
Extract Sparse Feature Vectors
Train MLP
Calculate Our Router Choices
Calculate Route LLM Choices
Calculate Result Stats