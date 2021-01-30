import matplotlib.pyplot as plt
import numpy as np

sensors = {
    'sensor1': {
        'rec1': (0, 10),
        'rec2': (13, 15),
    },
    'sensor2': {
        'rec1': (5, 8),
    }
}

yy = 0

color=iter(plt.cm.rainbow(np.linspace(0,1,len(sensors))))
#for i in range(n):
 #  c=next(color)
  # plt.plot(x, y,c=c)


fig, ax = plt.subplots(1,1)
for sensor_name, recording in sensors.items():
    
    c = next(color)
    for recording_number, (start, end) in  recording.items():
        
        print(sensor_name, recording_number, start, end)
        ax.plot((start,end),(yy,yy), color=c)
    yy += 1
    
    
ax.set_yticks(range(len(sensors)))
ax.set_yticklabels(sensors.keys())#f'{sensor_name}')

