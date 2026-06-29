# Database layer

Empty by design. When you add Postgres for case metadata, put SQLAlchemy models
and Alembic migrations here. The prediction path does not need a database — it
loads a model artifact — so this stays optional until you build case storage,
user accounts, or audit logging.
