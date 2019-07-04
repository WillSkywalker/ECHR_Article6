#!/usr/bin/env python3
# -_- coding: utf-8 -_-

from flask import Flask, request, jsonify

from flask_script import Manager
# from flask_migrate import Migrate, MigrateCommand
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy

import json

from .config import Config
from .models import metadata, CommunicatedCases, Decisions, Judgments

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker



app = Flask(__name__)
app.config.from_object(Config)
manager = Manager(app)
CORS(app)
db = SQLAlchemy(metadata=metadata)
db.init_app(app)

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=True)
Session = sessionmaker(bind=engine)

# create a Session
session = Session()


@app.route('/api/case/<appno>')
@cross_origin()
def word(appno):
    appno = appno[:-2] + '/' + appno[-2:]
    return str(session.query(Decisions).filter(Decisions.c.appno==appno).first().text)






# if __name__ == '__main__':
#     manager.run()
