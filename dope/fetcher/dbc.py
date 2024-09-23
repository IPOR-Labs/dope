import pathlib
import configparser
import pandas as pd
from sqlalchemy import create_engine


class PostgresConnection:
    def __init__(self, db_url):
        """
        Initialize the database connection.
        :param db_url: Database URL in the form of:
                        postgresql://username:password@host:port/database
        """
        self.db_url = db_url
        self.engine = None

    def __enter__(self):
        """Enter the runtime context related to this object."""
        try:
            self.engine = create_engine(self.db_url)
            # print("Connection to PostgreSQL database established.")
            return self
        except Exception as e:
            print(f"Error connecting to PostgreSQL database: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and close the database connection."""
        if self.engine:
            self.engine.dispose()
            # print("Connection to PostgreSQL database closed.")


class DataBaseConnector:
    def __init__(
        self,
        dbname,
        cred_file="dbs.ini",
        cred_folder=f"{pathlib.Path.home()}/credentials",
    ):
        self.dbname = dbname
        self.cred_file = cred_file
        self.cred_folder = cred_folder
        self.connection_string = self.get_connection_string()

    def get_connection_string(self):
        config = configparser.ConfigParser()
        config.read(f"{self.cred_folder}/{self.cred_file}")
        dbname = self.dbname
        username = config[dbname]["username"]
        pwd = config[dbname]["pwd"]
        port = config[dbname]["port"]
        host = config[dbname]["host"]
        database = config[dbname]["database"]
        return f"postgresql://{username}:{pwd}@{host}:{port}/{database}"

    def all_tables(self):
        q = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE' AND
          table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY table_schema, table_name;

    """
        return self.query(q)

    def query(self, query):
        with PostgresConnection(self.connection_string) as conn:
            try:
                df = pd.read_sql(query, conn.engine)
                return df
            except Exception as e:
                print(f"Error executing query: {e}")
                raise
