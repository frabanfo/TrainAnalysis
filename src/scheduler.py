from dramatiq_task import process_train, fetch_weather, process_data

if __name__ == "__main__":
    # Send messages to queues
    process_train.send(123)
    fetch_weather.send("LON-STN")
    process_data.send({"speed": 90, "temp": 22})

    print("Tasks dispatched! Start worker with:")
    print("  dramatiq tasks.train_tasks tasks.weather_tasks tasks.processing_tasks")
