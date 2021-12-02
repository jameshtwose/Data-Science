from sqlalchemy import select, Column, Integer, String, Sequence
from Backend.sql_utils.database import Base
from sqlalchemy.dialects.mysql import \
        BIGINT, BINARY, BIT, BLOB, BOOLEAN, CHAR, DATE, \
        DATETIME, DECIMAL, DECIMAL, DOUBLE, ENUM, FLOAT, INTEGER, \
        LONGBLOB, LONGTEXT, MEDIUMBLOB, MEDIUMINT, MEDIUMTEXT, NCHAR, \
        NUMERIC, NVARCHAR, REAL, SET, SMALLINT, TEXT, TIME, TIMESTAMP, \
        TINYBLOB, TINYINT, TINYTEXT, VARBINARY, VARCHAR, YEAR


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    name = Column(String(50))
    fullname = Column(String(50))
    nickname = Column(String(50))

    def __repr__(self):
        return "<User(name='%s', fullname='%s', nickname='%s')>" % (
                                self.name, self.fullname, self.nickname)


class Iris(Base):
    __tablename__ = "iris"
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    sepal_length = Column(FLOAT, nullable=False)
    sepal_width = Column(FLOAT, nullable=False)
    petal_length = Column(FLOAT, nullable=False)
    petal_width = Column(FLOAT, nullable=False)
    species = Column(String(50), nullable=False)

    def __repr__(self):
        return "<Iris(id='%s', sepal_length='%s', sepal_width='%s', petal_length='%s', petal_width='%s', species='%s')>"\
               % (self.id, self.sepal_length, self.sepal_width, self.petal_length, self.petal_width, self.species)