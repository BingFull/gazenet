import csv

from datetime import datetime as dt

from csv_utils import read_key_value_file

csv_path = 'gaze_positions.csv'
info_path = 'info.csv'
dst_path = 'gaze_positions_convert.csv'

def read_info(info_path):
    with open(info_path, 'r') as info_fh:
        info = read_key_value_file(info_fh)
    synced = float(info["Start Time (Synced)"])
    system = float(info["Start Time (System)"])
    return synced, system

def read_write_csv(csv_path, dst_path, synced_t, system_t):
    read_csvfile = open(csv_path, "r")
    reader = csv.reader(read_csvfile)
    rows = [row for row in reader]

    write_csvfile = open(dst_path, 'w')
    writer = csv.writer(write_csvfile)
    for row in rows:
        if row[0] == 'world_timestamp':
            continue
        else:
            tmp_timestamps = float(row[0])
            clock_t = clock_time(synced_t, system_t, tmp_timestamps)
            #date_t = datetime(clock_t+8*3600)
            row[0] = clock_t
            writer.writerow(row)
    read_csvfile.close()
    write_csvfile.close()


def clock_time(synced_time, system_time, pupil_time):
    return pupil_time - synced_time + system_time

def datetime(time_in_seconds):
    return dt.utcfromtimestamp(time_in_seconds).strftime('%Y-%m-%d %H:%M:%S.%f')


if __name__ == "__main__":
    synced_t, system_t = read_info(info_path)
    read_write_csv(csv_path, dst_path, synced_t, system_t)

