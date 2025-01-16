def _check_robot_position(x: int, y: int, robot_in_room) -> bool:
    x_in_room = (robot_in_room["left"] >= x) and (robot_in_room["right"] <= x)
    y_in_room = (robot_in_room["up"] >= y) and (robot_in_room["down"] <= y)
    return x_in_room and y_in_room


def check_robot_position(x: int, y: int, robot_in_room) -> bool:
    return (
        robot_in_room["left"] >= x >= robot_in_room["right"]
        and robot_in_room["up"] >= y >= robot_in_room["down"]
    )


robot_in_room_2 = {"left": 27250, "right": 22650, "up": 29600, "down": 24300}
robot_in_room_4 = {"left": 23900, "right": 20000, "up": 27350, "down": 25550}

print(_check_robot_position(25770, 25620, robot_in_room_4))
print(check_robot_position(22980, 26220, robot_in_room_4))
