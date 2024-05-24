from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

# 创建 ORM 模型的基类
Base = declarative_base()

# 定义 Log 模型
class Log(Base):
    __tablename__ = 'log'
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)

# 定义 Frame 模型
class Frame(Base):
    __tablename__ = 'frame'
    id = Column(Integer, primary_key=True)
    log_id = Column(Integer, ForeignKey('log.id'))
    log = relationship("Log", back_populates="frames")

# 为 Log 模型添加关系
Log.frames = relationship("Frame", order_by=Frame.id, back_populates="log")

# 定义 Config 模型
class Config(Base):
    __tablename__ = 'config'
    k = Column(String, primary_key=True)
    v = Column(String)

# 创建 SQLite 数据库和会话
engine = create_engine('sqlite:///client.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Log 表的 CRUD 操作
def create_log(start_time, end_time):
    new_log = Log(start_time=start_time, end_time=end_time)
    session.add(new_log)
    session.commit()
    return new_log

def get_log(log_id):
    return session.query(Log).filter_by(id=log_id).first()

def update_log(log_id, start_time=None, end_time=None):
    log = get_log(log_id)
    if log:
        if start_time:
            log.start_time = start_time
        if end_time:
            log.end_time = end_time
        session.commit()
    return log

def delete_log(log_id):
    log = get_log(log_id)
    if log:
        session.delete(log)
        session.commit()

# Frame 表的 CRUD 操作
def create_frame(log_id):
    new_frame = Frame(log_id=log_id)
    session.add(new_frame)
    session.commit()
    return new_frame

def get_frame(frame_id):
    return session.query(Frame).filter_by(id=frame_id).first()

def update_frame(frame_id, log_id=None):
    frame = get_frame(frame_id)
    if frame and log_id:
        frame.log_id = log_id
        session.commit()
    return frame

def delete_frame(frame_id):
    frame = get_frame(frame_id)
    if frame:
        session.delete(frame)
        session.commit()

# Config 表的 CRUD 操作
def create_config(k, v):
    new_config = Config(k=k, v=v)
    session.add(new_config)
    session.commit()
    return new_config

def get_config(k):
    config = session.query(Config).filter_by(k=k).first()
    if config:
        return config.v
    return None

def update_config(k, v):
    config = session.query(Config).filter_by(k=k).first()
    if config:
        config.v = v
        session.commit()
    return config

def delete_config(k):
    config = session.query(Config).filter_by(k=k).first()
    if config:
        session.delete(config)
        session.commit()

# 保存或更新 Config 表的项
def save_config(k, v):
    config = get_config(k)
    if config:
        # 如果配置项存在，更新值
        update_config(k, v)
    else:
        # 如果配置项不存在，创建新的配置项
        create_config(k, v)

if __name__ == "__main__":
    save_config("protect_type","face,number_plate")
