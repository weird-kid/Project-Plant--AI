from gpiozero import MCP3208
import keyboard

moisture = MCP3208(channel=1)
print('Starting Program',end="")

for i in range(10):
	print('.',end="")
	sleep(0.3)

print('Hold "s" key to print values\n Hold "q" key to stop program')

while keyboard.is_pressed('s'):

	print(f'Mositure -> {moisture.value} %')
	if keyboard.is_pressed('q'):
		break
	sleep(0.3)

