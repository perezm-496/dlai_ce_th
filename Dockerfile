FROM jupyter/scipy-notebook

# Create a directory for notebooks in the container
RUN mkdir -p /home/jovyan/work

# Copy the contents of the local /notebooks folder into the container
COPY notebooks /home/jovyan/work

# Set the working directory
WORKDIR /home/jovyan/work

# Expose the Jupyter Notebook server port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]