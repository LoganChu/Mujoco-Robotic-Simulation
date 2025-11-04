import mujoco
import mujoco.viewer

xml_path = "franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# This blocks until you close the window - simplest option!
mujoco.viewer.launch(model, data)