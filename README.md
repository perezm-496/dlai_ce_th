# Sample Jupyter Based Exercise

This project consists of a sample Jupyter notebook (`assignment.ipynb`) and a Python module (`assignment_test.py`) that evaluates and provides feedback for the task described in the notebook. To simplify its use, a `Dockerfile` is included.

## Installation

To set up the project locally, follow these instructions:

1. Clone the repository:

   \```   
   git clone https://github.com/perezm-496/dlai_ce_th.git
   cd dlai_ce_th
   \```

2. Ensure that Docker is installed on your machine. Build the Docker image:

   \```
   docker build -t sample-jupyter-exercise .
   \```


3. Run the Docker container:

   \```
   docker run -p 8888:8888 sample-jupyter-exercise
   \```

4. Open your web browser and navigate to `http://localhost:8888` to access the Jupyter Notebook interface.

## Usage

Open the assignment notebook (`assignment.ipynb`) in Jupyter Notebook and follow the instructions provided inside. You can write your solution and use the test section within the notebook to try out your solution. The test function also offers feedback if something goes wrong with your solution.

## Features

1. **Solution Notebook**: A complete solution is available in the `solution.ipynb` notebook.
2. **Feedback and Testing Code**: The code responsible for generating feedback and testing is located in `assignment_test.py`.
3. **Draft Notebook**: The `draft_assignment.ipynb` notebook was used during the creation of the assignment. This notebook contains some tests of the tests and the feedback provided by the `assignment_test.py` module.

## Contact Information

For questions or feedback, please contact peremz496@gmail.com.
