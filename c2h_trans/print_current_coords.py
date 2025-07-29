from pymycobot.elephantrobot import ElephantRobot


elephant_client = ElephantRobot("192.168.1.159", 5001)
elephant_client.start_client()
print(elephant_client.get_coords())

#coords=[  448 ,303,249, -96,73,-90]
#speed = 500

#elephant_client.write_coords(coords,speed)