# database.py
# Everything about connecting Python to our PostgreSQL database lives here.
# Other files will import `engine` from this file instead of connecting themselves.

import os                      # lets us read environment variables
from pathlib import Path       # a clean way to work with file/folder paths

from dotenv import load_dotenv           # reads a .env file into environment variables
from sqlalchemy import create_engine, text

# Our .env file lives in the project ROOT, one folder above this file (backend/).
# __file__ = the path of this file; .parent = its folder (backend/); .parent.parent = project root.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Read the .env file: every NAME=value line becomes an environment variable
# that os.environ can see. This is how the secret settings reach our code.
load_dotenv(ENV_FILE)

# Look up the database address. os.environ["..."] raises an error if it is missing,
# which is what we want: fail loudly at startup instead of mysteriously later.
DATABASE_URL: str = os.environ["DATABASE_URL"]

# The ENGINE is SQLAlchemy's connection manager. It does not connect yet;
# it just knows the address and opens (and reuses) connections when asked.
# Think of it as the switchboard that keeps a few phone lines open to the database.
engine = create_engine(DATABASE_URL)


def check_connection() -> str:
    """Ask the database for its version string, to prove the connection works."""
    # `with` opens a connection and guarantees it is closed again afterwards.
    with engine.connect() as connection:
        # text() marks our string as raw SQL. .scalar() returns the single value.
        version = connection.execute(text("SELECT version();")).scalar()
    return str(version)


# This block only runs when we execute THIS file directly (python database.py),
# not when another file imports it. Handy for quick manual tests.
if __name__ == "__main__":
    print(check_connection())
