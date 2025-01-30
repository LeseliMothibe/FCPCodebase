def main():
    from scripts.commons.Script import Script
    script = Script()  # Initialize script

    # Allow using local version of StableBaselines3
    import sys
    from os.path import dirname, abspath, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "stable-baselines3"))

    from scripts.commons.UI import UI
    from os.path import isfile, join, realpath, dirname
    from os import listdir, getcwd
    from importlib import import_module

    _cwd = realpath(join(getcwd(), dirname(__file__)))
    gyms_path = _cwd + "/scripts/gyms/"
    utils_path = _cwd + "/scripts/utils/"
    exclusions = ["__init__.py"]

    utils = sorted([f[:-3] for f in listdir(utils_path) if isfile(join(utils_path, f)) and f.endswith(".py") and f not in exclusions], key=lambda x: (x != "Server", x))
    gyms = sorted([f[:-3] for f in listdir(gyms_path) if isfile(join(gyms_path, f)) and f.endswith(".py") and f not in exclusions])

    while True:
        _, col_idx, col = UI.print_table([utils, gyms], ["Demos & Tests & Utils", "Gyms"], cols_per_title=[2, 1], numbering=[True] * 2, prompt='Choose script (ctrl+c to exit): ')

        is_gym = False
        if col == 0:
            chosen = ("scripts.utils.", utils[col_idx])
        elif col == 1:
            chosen = ("scripts.gyms.", gyms[col_idx])
            is_gym = True

        cls_name = chosen[1]
        mod = import_module(chosen[0] + chosen[1])

        if not is_gym:
            # Execute utility scripts
            from world.commons.Draw import Draw
            from agent.Base_Agent import Base_Agent
            obj = getattr(mod, cls_name)(script)
            try:
                obj.execute()
            except KeyboardInterrupt:
                print("\nctrl+c pressed, returning...\n")
            Draw.clear_all()
            Base_Agent.terminate_all()
            script.players = []
            del obj
        else:
            # Execute gym scripts
            from scripts.commons.Train_Base import Train_Base

            print("\nBefore using GYMS, make sure all server parameters are set correctly")
            print("(sync mode should be 'On', real time should be 'Off', cheats should be 'On', ...)")
            print("To change these parameters go to the previous menu, and select Server\n")
            print("Also, GYMS start their own servers, so don't run any server manually")

            while True:
                try:
                    idx = UI.print_table([["Train Running", "Train Stopping", "Test"]], numbering=[True], prompt='Choose option (ctrl+c to return): ')[0]
                except KeyboardInterrupt:
                    print()
                    break

                if idx == 0:
                    # Train running behavior
                    mod.Train(script).train_running(dict())
                elif idx == 1:
                    # Train stopping behavior
                    model_info = Train_Base.prompt_user_for_model()
                    if model_info is not None:
                        mod.Train(script).train_stopping({"running_model_path": model_info["model_file"]})
                elif idx == 2:
                    # Test behavior
                    model_info = Train_Base.prompt_user_for_model()
                    if model_info is not None:
                        mod.Train(script).test(model_info)