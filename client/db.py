import os
import json
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import text
from databases import Database
from sqlalchemy import create_engine, MetaData
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
    base64 = Column(String)
    time = Column(DateTime)
    data = Column(String)  # 存储JSON数据
    log = relationship("Log", back_populates="frames")

# 为 Log 模型添加关系
Log.frames = relationship("Frame", order_by=Frame.id, back_populates="log")

# 定义 Config 模型
class Config(Base):
    __tablename__ = 'config'
    k = Column(String, primary_key=True)
    v = Column(String)

# 获取当前脚本文件的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 创建 SQLite 数据库和会话
path = f'sqlite:///{os.path.join(script_dir, "client.db")}'
engine = create_engine(path)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
database = Database(path)
# Log 表的 CRUD 操作
def create_log(start_time, end_time):
    new_log = Log(start_time=start_time, end_time=end_time)
    session.add(new_log)
    session.commit()
    return new_log
def get_all_logs():
    return session.query(Log).all()

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
        # 删除所有frame记录
        for frame in log.frames:
            session.delete(frame)
        session.delete(log)
        session.commit()

def get_frames_by_log_id(log_id: int = 0):
    query = session.query(Frame)
    if log_id!=0:
        query = query.filter(Frame.log_id == log_id)
    return query

# Frame 表的 CRUD 操作
def create_frame(log_id, time, data=None, base64=None):
    if data:
        data = json.dumps(data)
    new_frame = Frame(log_id=log_id,  time=time, data=data,base64=base64)  # 序列化JSON数据
    session.add(new_frame)
    session.commit()
    return new_frame

def get_frame(frame_id):
    frame = session.query(Frame).filter_by(id=frame_id).first()
    if frame:
        frame.data = json.loads(frame.data)  # 反序列化JSON数据
    return frame

def update_frame(frame_id, log_id=None, base64=None, time=None, data=None):
    frame = get_frame(frame_id)
    if frame:
        if log_id:
            frame.log_id = log_id
        if base64:
            frame.base64 = base64
        if time:
            frame.time = time
        if data:
            frame.data = json.dumps(data)  # 序列化JSON数据
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

def clear_all_logs_and_frames():
    try:
        # 删除所有的 frame 和 log 记录
        session.query(Frame).delete()
        session.query(Log).delete()
        session.commit()
        
        # 重置 Log 表的主键自增计数
        session.execute(text("DELETE FROM sqlite_sequence WHERE name='log'"))
        session.commit()
        
        # 重置 Frame 表的主键自增计数
        session.execute(text("DELETE FROM sqlite_sequence WHERE name='frame'"))
        session.commit()
        # 执行VACUUM
        session.execute(text("VACUUM"))
        session.commit()
        print("All logs and frames have been cleared, and the primary key sequences have been reset.")
    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")

# 调用函数清除所有log和frame的记录
if __name__ == "__main__":
    clear_all_logs_and_frames()
