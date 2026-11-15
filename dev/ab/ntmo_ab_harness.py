"""Neural Tone Mapper - same-camera A/B harness (file-handshake version).

Both captures must come from ONE editor session with the camera pinned, otherwise the
camera, sun angle and streaming state differ and a per-pixel comparison against
Unreal's tonemapper is meaningless.

The earlier frame-count version failed for a timing reason worth recording: it printed
ON_READY and then flipped the plugin back off a fixed number of frames later, while the
external driver needed ~1 s to grab the window. Both screenshots therefore ended up
taken with the plugin already disabled, and the A/B honestly reported "identical
images". The sequence is now driven by the driver through sentinel files instead of a
timer.

Handshake (all under E:/ConveyanceSpline):
    driver -> harness : _ab_step1.flag     "plugin ON captured, you may disable"
                        _ab_step2.flag     "plugin OFF captured, you may finish"
    harness -> driver : log markers [NTMOAB] ON_READY / OFF_READY / DONE
"""

import os

import unreal

ROOT = r"E:\ConveyanceSpline"
STEP1 = os.path.join(ROOT, "_ab_step1.flag")
STEP2 = os.path.join(ROOT, "_ab_step2.flag")

# Fixed three-quarter view of the temple: deep shadow, a bright direct-lit face and a
# saturated backdrop, so tone curve differences actually show instead of hiding in a
# flat grey frame.
CAMERA_LOCATION = unreal.Vector(-900.0, -900.0, 420.0)
CAMERA_ROTATION = unreal.Rotator(pitch=-12.0, yaw=45.0, roll=0.0)

_state = {
    "frame": 0, "handle": None, "phase": "wait_camera", "phase_frame": 0,
    "perf_sum": 0.0, "perf_n": 0,
}

# Frame-time measurement. Both the editor UI and the viewport are identical between the
# two windows, so the delta isolates what the plugin costs. vsync and the frame cap must
# be off or the measurement just reads 16.7 ms twice.
PERF_FRAMES = 400


def _perf_reset():
    _state["perf_sum"] = 0.0
    _state["perf_n"] = 0


def _perf_accumulate(dt):
    _state["perf_sum"] += dt
    _state["perf_n"] += 1


def _perf_report(tag):
    n = max(_state["perf_n"], 1)
    avg_ms = 1000.0 * _state["perf_sum"] / n
    _say("[NTMOAB] PERF_{} avg_ms={:.3f} frames={}".format(tag, avg_ms, n))
    return avg_ms


def _say(msg):
    print(msg)


def _set_camera():
    try:
        sub = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        sub.set_level_viewport_camera_info(CAMERA_LOCATION, CAMERA_ROTATION, "")
        loc, rot = sub.get_level_viewport_camera_info("")
        _say("[NTMOAB] CAMERA_READY loc={} rot={}".format(loc, rot))
        return
    except Exception as exc:
        _say("[NTMOAB] camera via subsystem failed: {}".format(exc))

    try:
        unreal.EditorLevelLibrary.set_level_viewport_camera_info(
            CAMERA_LOCATION, CAMERA_ROTATION, "")
        _say("[NTMOAB] CAMERA_READY via EditorLevelLibrary")
    except Exception as exc:
        _say("[NTMOAB] CAMERA_FAILED {}".format(exc))


def _cvar(name, value):
    try:
        unreal.SystemLibrary.execute_console_command(None, "{} {}".format(name, value))
        return True
    except Exception as exc:
        _say("[NTMOAB] cvar {} failed: {}".format(name, exc))
        return False


def _frames_in_phase():
    return _state["frame"] - _state["phase_frame"]


def _enter(phase):
    _state["phase"] = phase
    _state["phase_frame"] = _state["frame"]


def _consume(path):
    if not os.path.exists(path):
        return False
    try:
        os.remove(path)
    except OSError:
        pass
    return True


def _tick(delta_seconds):
    _state["frame"] += 1
    phase = _state["phase"]

    if phase == "wait_camera":
        # Let the level stream in before pinning or capturing anything; a half-loaded
        # frame is the classic way to poison an A/B.
        if _frames_in_phase() > 90:
            _set_camera()
            _cvar("r.NTMO.Enable", 1)
            _enter("settle_on")

    elif phase == "settle_on":
        # TAA history and auto exposure need a moment to converge after the camera jump
        # and the tonemapper swap, or the two captures differ for the wrong reason.
        if _frames_in_phase() > 90:
            _perf_reset()
            _enter("measure_on")
        else:
            _perf_accumulate(delta_seconds)

    elif phase == "measure_on":
        _perf_accumulate(delta_seconds)
        if _frames_in_phase() > PERF_FRAMES:
            _perf_report("ON")
            _say("[NTMOAB] ON_READY")
            _enter("wait_step1")

    elif phase == "wait_step1":
        if _consume(STEP1):
            _cvar("r.NTMO.Enable", 0)
            _enter("settle_off")

    elif phase == "settle_off":
        if _frames_in_phase() > 60:
            _perf_reset()
            _enter("measure_off")
        else:
            _perf_accumulate(delta_seconds)

    elif phase == "measure_off":
        _perf_accumulate(delta_seconds)
        if _frames_in_phase() > PERF_FRAMES:
            _perf_report("OFF")
            _say("[NTMOAB] OFF_READY")
            _enter("wait_step2")

    elif phase == "wait_step2":
        if _consume(STEP2):
            _say("[NTMOAB] DONE")
            _enter("done")
            try:
                unreal.unregister_slate_post_tick_callback(_state["handle"])
            except Exception:
                pass


def _start():
    # Uncapped frame rate is required for the frame-time comparison to mean anything.
    _cvar("t.MaxFPS", 0)
    _cvar("r.VSync", 0)
    for p in (STEP1, STEP2):
        _consume(p)
    _state["handle"] = unreal.register_slate_post_tick_callback(_tick)
    _say("[NTMOAB] harness registered (handshake mode)")


_start()
