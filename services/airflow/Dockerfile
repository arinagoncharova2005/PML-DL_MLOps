FROM apache/airflow:2.10.2

ENV PROJECTPATH=/pmldlapp
ADD requirements.txt .
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt
