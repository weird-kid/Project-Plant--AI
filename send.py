import requests
import os
from .aht10_temp import *
from .soil_moist import *
import random

def send():    
    url = "http://172.16.18.178:5000/index/"
    file_path1 = "/home/dhruv/PlantAi/sensors/photos/plant1.jpg"
    
    no = random.randrange(1,10)
    file_path2 = ".jpg"
    file_path = file_path1 + no.strip() + file_path2
    
    '''dabsp=os.path.abspath(file_path)
    print(dabsp)
    '''
    
    data = {
        "plantname": "tulasi",  
        "humidity": 65,
        "temperature": 45,
        "moisture": 70
    }
    
    
    data['humidity'] , data['temperature'] = main()
    data['mositure'] = Temp_humid()
    
    print('')
    print('')
    print(f'humidity->{data["humidity"]}')
    print(f'Temperature->{data["temperature"]}')
    print(f'Mositure->{data["mositure"]}')
    print('')
    print('')
    
    with open(file_path, "rb") as file:
        files = {"image": file}
        response = requests.post(url, data=data, files=files)
    
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except requests.exceptions.JSONDecodeError:
        print("Response Text:", response.text)
