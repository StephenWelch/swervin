import argparse
import time
import mujoco
import mujoco.viewer


def main(args):
    model = mujoco.MjModel.from_xml_path("assets/models/reduced/mjcf/scene.xml")
    data = mujoco.MjData(model)

    def key_cb(keycode: int):
        if chr(keycode) == 'R':
            gantry_eq_id = model.equality("gantry").id
            if data.eq_active[gantry_eq_id] == 1:
                data.eq_active[gantry_eq_id] = 0
            else:
                data.eq_active[gantry_eq_id] = 1
                # mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
                print("Releasing gantry")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        while True:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1/60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)