from isaacsim.sensors.camera import Camera

def create_camera():
    camera = Camera(
        prim_path="/World/Camera",
        position=[1.2, 0.0, 1.0],
        frequency=10,
        resolution=(256, 256),
    )
    camera.initialize()
    return camera