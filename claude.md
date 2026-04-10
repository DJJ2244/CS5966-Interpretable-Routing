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
- DAOs are the only place where queries are ran on the datbase