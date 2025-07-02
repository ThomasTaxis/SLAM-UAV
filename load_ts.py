from datetime import datetime, timedelta
import os



def load_timestamps(ts_dir, numeric=False):
    ts = []
    file_path = os.path.join(ts_dir, 'timestamps.txt')
    
    try:
        with open(file_path, 'r') as file:
            ts = [line.strip() for line in file]
            
        if numeric:
            new_ts = []
            for time_str in ts:
                # Split the timestamp into the main part and the decimal part of seconds
                main_part, decimal_part = time_str.split('.')
                # Convert the main part to a datetime object
                # Adjust the format as needed based on your timestamps
                dt = datetime.strptime(main_part, "%Y-%m-%d %H:%M:%S")
                # Add the decimal part of seconds
                decimal_seconds = timedelta(seconds=float('0.' + decimal_part))
                final_timestamp = dt + decimal_seconds
                # Convert to Unix timestamp and include the decimal part
                new_ts.append(final_timestamp.timestamp())
            ts = new_ts
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return ts  