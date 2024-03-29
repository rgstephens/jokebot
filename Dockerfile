FROM rasa/rasa-sdk:3.3.0
ENV TZ=Americas/Los_Angeles

COPY actions /app/actions

USER root
RUN /opt/venv/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/actions/requirements-actions.txt

USER 1001
CMD ["start", "--actions", "actions"]