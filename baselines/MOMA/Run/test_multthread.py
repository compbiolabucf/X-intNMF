import multiprocessing
import time

# Define a function to run a model
def run_model(model_name, result_queue):
    # Simulate model execution
    print(f"Starting model: {model_name}")
    time.sleep(3)  # Replace with actual model execution
    result = f"Result from {model_name}"
    
    # Put result in the queue
    result_queue.put(result)
    print(f"Finished model: {model_name}")

if __name__ == "__main__":
    # List of model names (or any identifier)
    models = ["Model_1", "Model_2", "Model_3", "Model_4"]

    # Queue to store results
    result_queue = multiprocessing.Queue()

    # List to store process objects
    processes = []

    # Create and start a process for each model
    for model in models:
        process = multiprocessing.Process(target=run_model, args=(model, result_queue))
        process.start()
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Collect results from the queue
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Print all results
    print("All results:", results)
