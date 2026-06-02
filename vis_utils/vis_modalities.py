import os
import argparse
import numpy as np
import zarr
import imageio.v2 as imageio

# Optional, only used for tacff image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Basic utils
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_zarr_frames(arr, frame_ids):
    # zarr basic indexing does not support numpy array fancy indexing
    return np.stack([np.asarray(arr[int(i)]) for i in frame_ids], axis=0)


def normalize_rgb_image(img):
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)

    if np.nanmax(img) <= 1.0:
        img = img * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


def save_rgb_frames(zarr_path, out_dir, frame_ids, name="rgb"):
    if zarr_path is None:
        return

    arr = zarr.open(zarr_path, mode="r")
    print(f"\n[{name}] path: {zarr_path}")
    print(f"[{name}] shape: {arr.shape}, dtype: {arr.dtype}")

    ensure_dir(out_dir)

    for local_i, global_i in enumerate(frame_ids):
        img = normalize_rgb_image(arr[int(global_i)])
        imageio.imwrite(os.path.join(out_dir, f"{local_i:04d}_t{int(global_i):06d}.png"), img)

    print(f"[✓] saved {name} images to: {out_dir}")


def save_depth_frames(
    zarr_path,
    out_dir,
    frame_ids,
    invert=False,
    global_normalize=True,
    bg_value=0.0,
    smooth_sigma=5.0,
    alpha_sigma=None,
    gamma=0.6,
    alpha_gamma=0.8,
    cool_tone=True,
):
    """
    Soft tactile depth visualization:
        black background + smooth white deformation

    Key improvement:
        use continuous depth alpha instead of binary mask alpha,
        so the boundary is much smoother and less jagged.
    """

    import os
    import cv2
    import zarr
    import imageio
    import numpy as np

    if zarr_path is None:
        return

    if alpha_sigma is None:
        alpha_sigma = smooth_sigma * 1.5

    arr = zarr.open(zarr_path, mode="r")

    print(f"\n[tacdepth] path: {zarr_path}")
    print(f"[tacdepth] shape: {arr.shape}, dtype: {arr.dtype}")

    os.makedirs(out_dir, exist_ok=True)

    depths = []
    for global_i in frame_ids:
        depth = np.asarray(arr[int(global_i)], dtype=np.float32)

        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        depth = np.clip(depth, 0.0, 1.0)
        depths.append(depth)

    if len(depths) == 0:
        print("[WARN] no depth frames")
        return

    if global_normalize:
        valid_vals = []

        for d in depths:
            valid = d[d > bg_value + 1e-6]
            if valid.size > 0:
                valid_vals.append(valid)

        if len(valid_vals) > 0:
            all_valid = np.concatenate(valid_vals, axis=0)
            d_min = float(np.percentile(all_valid, 2))
            d_max = float(np.percentile(all_valid, 98))
        else:
            d_min, d_max = 0.0, 1.0

        if d_max <= d_min + 1e-8:
            d_min, d_max = 0.0, 1.0

        print(f"[tacdepth] global range: {d_min:.6f} ~ {d_max:.6f}")

    for local_i, global_i in enumerate(frame_ids):
        depth = depths[local_i]

        # Smooth raw depth for visual intensity
        if smooth_sigma is not None and smooth_sigma > 0:
            depth_smooth = cv2.GaussianBlur(
                depth,
                ksize=(0, 0),
                sigmaX=smooth_sigma,
                sigmaY=smooth_sigma,
            )
        else:
            depth_smooth = depth

        # Normalize
        if global_normalize:
            norm = (depth_smooth - d_min) / (d_max - d_min + 1e-8)
        else:
            valid = depth_smooth[depth_smooth > bg_value + 1e-6]
            if valid.size > 0:
                d_min_i = float(np.percentile(valid, 2))
                d_max_i = float(np.percentile(valid, 98))
            else:
                d_min_i, d_max_i = 0.0, 1.0

            if d_max_i <= d_min_i + 1e-8:
                d_min_i, d_max_i = 0.0, 1.0

            norm = (depth_smooth - d_min_i) / (d_max_i - d_min_i + 1e-8)

        norm = np.clip(norm, 0.0, 1.0)

        if invert:
            norm = 1.0 - norm

        # Brightness shaping
        intensity_norm = norm ** gamma

        # Continuous alpha, not binary mask
        alpha = norm.copy()

        if alpha_sigma is not None and alpha_sigma > 0:
            alpha = cv2.GaussianBlur(
                alpha,
                ksize=(0, 0),
                sigmaX=alpha_sigma,
                sigmaY=alpha_sigma,
            )

        alpha = np.clip(alpha, 0.0, 1.0)
        alpha = alpha ** alpha_gamma

        # Build soft white / cool white RGB
        intensity = intensity_norm * 255.0

        if cool_tone:
            r = intensity * 0.92
            g = intensity * 0.95
            b = intensity * 1.00
        else:
            r = intensity
            g = intensity
            b = intensity

        vis = np.stack([r, g, b], axis=-1)

        # Black background blend
        vis = vis * alpha[..., None]

        vis = np.clip(vis, 0, 255).astype(np.uint8)

        imageio.imwrite(
            os.path.join(out_dir, f"{local_i:04d}_t{int(global_i):06d}.png"),
            vis,
        )

    print(f"[✓] saved soft tactile depth images to: {out_dir}")


def save_tacff_frames(
    zarr_path,
    out_dir,
    frame_ids,
    resolution=40,
    arrow_scale=0.0008,
    use_global_scale=True,
    transpose_hw=True,
    contact_percentile=40,
):
    """
    Visualize tactile force field.

    Correct convention verified from data:
        tacff[..., 0] = fz / indentation / normal force
        tacff[..., 1] = fx
        tacff[..., 2] = fy

    Visualization:
        color  = fz, green -> red
        arrow  = (fx, fy)
        layout = transpose H/W for display
    """

    if zarr_path is None:
        return

    arr = zarr.open(zarr_path, mode="r")
    print(f"\n[tacff] path: {zarr_path}")
    print(f"[tacff] shape: {arr.shape}, dtype: {arr.dtype}")

    assert arr.ndim == 4 and arr.shape[-1] == 3, \
        f"Expected tacff shape [T,H,W,3], got {arr.shape}"

    ensure_dir(out_dir)

    data = read_zarr_frames(arr, frame_ids).astype(np.float32)

    # For global color normalization, use fz channel
    data_for_scale = np.transpose(data, (0, 2, 1, 3)) if transpose_hw else data
    fz_all = np.abs(data_for_scale[..., 0])

    if use_global_scale:
        normal_max_global = float(np.nanpercentile(fz_all, 98)) + 1e-8
    else:
        normal_max_global = None

    print("[tacff] channel_order: [fz, fx, fy]")
    print("[tacff] vector: original direction, quiver=(fx, fy)")
    print(f"[tacff] transpose_hw: {transpose_hw}")
    if use_global_scale:
        print(f"[tacff] global fz 98pct: {normal_max_global:.8f}")

    for local_i, global_i in enumerate(frame_ids):
        frame = data[local_i]

        if transpose_hw:
            frame = np.transpose(frame, (1, 0, 2))

        H, W = frame.shape[:2]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))

        # verified convention
        fz = np.abs(frame[..., 0])
        fx = frame[..., 1]
        fy = frame[..., 2]

        # only draw contact-ish area
        contact_thr = np.nanpercentile(fz, contact_percentile)
        mask = fz >= contact_thr

        fx_vis = fx.copy()
        fy_vis = fy.copy()
        fx_vis[~mask] = 0.0
        fy_vis[~mask] = 0.0

        if use_global_scale:
            normal_max = normal_max_global
        else:
            normal_max = float(np.nanpercentile(fz, 98)) + 1e-8

        contact = np.clip(fz / normal_max, 0.0, 1.0)

        colors = np.zeros((H, W, 4), dtype=np.float32)
        colors[..., 0] = contact
        colors[..., 1] = 1.0 - contact
        colors[..., 2] = 0.0
        colors[..., 3] = np.where(mask, 1.0, 0.25)

        target_aspect = 4.0 / 3.0   # width / height

        fig_h = H * resolution / 100
        fig_w = fig_h * target_aspect

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        # background taxel dots
        dot_colors = np.zeros((H, W, 4), dtype=np.float32)

        # use same green->red mapping as force
        dot_colors[..., 0] = contact * 0.7
        dot_colors[..., 1] = (1.0 - contact) * 0.95
        dot_colors[..., 2] = 0.0

        # make non-contact dots still visible
        dot_colors[..., 3] = 0.75

        ax.scatter(
            X,
            Y,
            s=30,          # bigger dots
            c=dot_colors.reshape(-1, 4),
            linewidths=0,
        )

        ax.quiver(
            X,
            Y,
            fx_vis,
            fy_vis,
            color=colors.reshape(-1, 4),
            angles="xy",
            scale_units="xy",
            scale=arrow_scale,

            width=0.008,

            headwidth=5.5,
            headlength=7.5,
            headaxislength=6.0,

            pivot="tail",
        )
        
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., :3]

        imageio.imwrite(
            os.path.join(out_dir, f"{local_i:04d}_t{int(global_i):06d}.png"),
            img,
        )

        plt.close(fig)

    print(f"[✓] saved tacff images to: {out_dir}")

# ============================================================
# Point cloud transform / export
# ============================================================

def compute_pc_lims_for_image(
    xyz,
    use_percentile=True,
    percentile=2.0,
    zoom=1.0,
):
    xyz = np.asarray(xyz)
    valid_xyz = xyz[np.isfinite(xyz).all(axis=-1)]

    if valid_xyz.size == 0:
        return (-1, 1), (-1, 1), (-1, 1)

    if use_percentile:
        xyz_min = np.percentile(valid_xyz, percentile, axis=0)
        xyz_max = np.percentile(valid_xyz, 100.0 - percentile, axis=0)
    else:
        xyz_min = np.nanmin(valid_xyz, axis=0)
        xyz_max = np.nanmax(valid_xyz, axis=0)

    center = (xyz_min + xyz_max) / 2.0
    max_range = np.max(xyz_max - xyz_min) / 2.0

    if max_range <= 1e-8:
        max_range = 1.0

    max_range = max_range * zoom

    return (
        (center[0] - max_range, center[0] + max_range),
        (center[1] - max_range, center[1] + max_range),
        (center[2] - max_range, center[2] + max_range),
    )


def render_single_pointcloud_image_matplotlib(
    pc,
    output_png,
    rx=60,
    ry=120,
    rz=0,
    elev=20,
    azim=-225,
    point_size=40,
    black_background=True,
    hide_axis=True,
    percentile=2.0,
    zoom=1.0,
    figsize=6,
    dpi=200,
):
    """
    Render one point cloud frame as black-background image.

    pc: [N,3] or [N,6]
        if [N,6], last 3 channels are RGB in [0,1] or [0,255].
    """
    pc = np.asarray(pc, dtype=np.float32)

    assert pc.ndim == 2 and pc.shape[1] in [3, 6], \
        f"Expected pointcloud [N,3] or [N,6], got {pc.shape}"

    xyz = pc[:, :3].copy()

    # use your existing euler_to_R / transform_xyz if already defined
    R = euler_to_R(rx, ry, rz)

    xyz = transform_xyz(
        xyz,
        R=R,
        t=np.zeros(3, dtype=np.float32),
        rotate_about_center=True,
    )

    pc_vis = pc.copy()
    pc_vis[:, :3] = xyz

    x_lim, y_lim, z_lim = compute_pc_lims_for_image(
        xyz,
        use_percentile=True,
        percentile=percentile,
        zoom=zoom,
    )

    pts = pc_vis[:, :3]
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]

    if pc_vis.shape[1] == 6:
        colors = pc_vis[:, 3:6][valid]
        if np.nanmax(colors) > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = "white"

    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    if black_background:
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=colors,
        s=point_size,
        depthshade=False,
        linewidths=0,
    )

    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    if hide_axis:
        ax.set_axis_off()
        ax.grid(False)
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if black_background:
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.zaxis.label.set_color("white")
            ax.tick_params(colors="white")

            ax.xaxis.set_pane_color((0, 0, 0, 1))
            ax.yaxis.set_pane_color((0, 0, 0, 1))
            ax.zaxis.set_pane_color((0, 0, 0, 1))

    try:
        ax.dist = 6
    except Exception:
        pass

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ensure_dir(os.path.dirname(output_png))
    plt.savefig(
        output_png,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(fig)

    print(f"[✓] saved pointcloud image: {output_png}")


def render_single_pointcloud_image_from_zarr(
    pc_zarr_path,
    output_png,
    frame=0,
    rx=60,
    ry=120,
    rz=0,
    elev=20,
    azim=-225,
    point_size=40,
    percentile=2.0,
    zoom=1.0,
    hide_axis=True,
):
    arr = zarr.open(pc_zarr_path, mode="r")

    print(f"\n[pointcloud image] path: {pc_zarr_path}")
    print(f"[pointcloud image] shape: {arr.shape}, dtype: {arr.dtype}")

    assert arr.ndim == 3 and arr.shape[-1] in [3, 6], \
        f"Expected pointcloud shape [T,N,3] or [T,N,6], got {arr.shape}"

    frame = int(frame)
    frame = max(0, min(frame, arr.shape[0] - 1))

    pc = np.asarray(arr[frame], dtype=np.float32)

    render_single_pointcloud_image_matplotlib(
        pc=pc,
        output_png=output_png,
        rx=rx,
        ry=ry,
        rz=rz,
        elev=elev,
        azim=azim,
        point_size=point_size,
        black_background=True,
        hide_axis=hide_axis,
        percentile=percentile,
        zoom=zoom,
    )


def euler_to_R(rx_deg=0, ry_deg=0, rz_deg=0):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ], dtype=np.float32)

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1],
    ], dtype=np.float32)

    return Rz @ Ry @ Rx


def transform_xyz(xyz, R=None, t=None, rotate_about_center=True):
    xyz = xyz.astype(np.float32)

    if R is None:
        R = np.eye(3, dtype=np.float32)

    if t is None:
        t = np.zeros(3, dtype=np.float32)

    valid = np.isfinite(xyz).all(axis=-1)

    if rotate_about_center:
        center = np.nanmean(xyz[valid], axis=0)
    else:
        center = np.zeros(3, dtype=np.float32)

    xyz_out = xyz.copy()
    xyz_out[valid] = (xyz[valid] - center) @ R.T + center + t
    return xyz_out


def export_pointcloud_npz(
    pc_zarr_path,
    out_npz,
    frame_ids,
    rx=60,
    ry=120,
    rz=0,
    tx=0,
    ty=0,
    tz=0,
):
    arr = zarr.open(pc_zarr_path, mode="r")
    print(f"\n[pointcloud] path: {pc_zarr_path}")
    print(f"[pointcloud] shape: {arr.shape}, dtype: {arr.dtype}")

    assert arr.ndim == 3 and arr.shape[-1] in [3, 6], \
        f"Expected pointcloud shape [T,N,3] or [T,N,6], got {arr.shape}"

    data = read_zarr_frames(arr, frame_ids).astype(np.float32)

    xyz = data[..., :3]
    rgb = data[..., 3:6] if data.shape[-1] == 6 else None

    R = euler_to_R(rx, ry, rz)
    t = np.array([tx, ty, tz], dtype=np.float32)

    xyz = transform_xyz(
        xyz,
        R=R,
        t=t,
        rotate_about_center=True,
    )

    if rgb is not None:
        if np.nanmax(rgb) > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    ensure_dir(os.path.dirname(out_npz))

    np.savez_compressed(
        out_npz,
        xyz=xyz.astype(np.float32),
        rgb=rgb.astype(np.float32) if rgb is not None else None,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
    )

    print(f"[✓] saved pointcloud npz: {out_npz}")
    print("xyz shape:", xyz.shape)
    print("rgb shape:", None if rgb is None else rgb.shape)


def build_frame_ids(start, end, stride, max_T=None):
    if end is None:
        if max_T is None:
            raise ValueError("end is None but max_T is also None.")
        end = max_T

    if max_T is not None:
        end = min(end, max_T)

    return np.arange(start, end, stride, dtype=np.int64)


def export_all(args):
    pc_arr = zarr.open(args.pc_zarr_path, mode="r")
    T = pc_arr.shape[0]

    frame_ids = build_frame_ids(
        start=args.start,
        end=args.end,
        stride=args.stride,
        max_T=T,
    )

    print("\n================ EXPORT ================")
    print("num frames:", len(frame_ids))
    print("frame ids:", frame_ids)

    export_pointcloud_npz(
        pc_zarr_path=args.pc_zarr_path,
        out_npz=args.npz_path,
        frame_ids=frame_ids,
        rx=args.rx,
        ry=args.ry,
        rz=args.rz,
        tx=args.tx,
        ty=args.ty,
        tz=args.tz,
    )

    base_dir = os.path.dirname(args.npz_path)

    save_rgb_frames(
        args.front_zarr_path,
        os.path.join(base_dir, "front"),
        frame_ids,
        name="front",
    )

    save_rgb_frames(
        args.wrist_zarr_path,
        os.path.join(base_dir, "wrist"),
        frame_ids,
        name="wrist",
    )

    save_rgb_frames(
        args.tacrgb_zarr_path,
        os.path.join(base_dir, "tactile_rgb_right"),
        frame_ids,
        name="tactile_rgb_right",
    )


    save_depth_frames(
        args.tacdepth_zarr_path,
        os.path.join(base_dir, "tactile_depth_right"),
        frame_ids,
        invert=False,
        global_normalize=True,
        smooth_sigma=1.0,
        gamma=0.0,
        alpha_gamma=0.5,
        cool_tone=True,
    )

    save_tacff_frames(
        args.tacff_zarr_path,
        os.path.join(base_dir, "tactile_force_field_right"),
        frame_ids,
        resolution=args.tacff_resolution,
        arrow_scale=args.tacff_arrow_scale,
        use_global_scale=True,
        transpose_hw=True,
        contact_percentile=40,
    )


# ============================================================
# Mitsuba render
# ============================================================

def render_npz_with_mitsuba(
    npz_path,
    out_dir,
    frames=(0,),
    sphere_radius=0.015,
    standardize=True,
    width=1920,
    height=1080,
    sample_count=128,
    camera_y=-3.0,
    camera_z=2.0,
    target_z=0.15,
    fov=20,
):
    import mitsuba as mi
    mi.set_variant("scalar_rgb")

    data = np.load(npz_path, allow_pickle=True)
    xyz_all = data["xyz"]
    rgb_all = data["rgb"]

    if rgb_all.shape == ():
        rgb_all = None

    ensure_dir(out_dir)

    for f in frames:
        xyz = xyz_all[f].copy()
        rgb = rgb_all[f].copy() if rgb_all is not None else None

        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]

        if rgb is not None:
            rgb = rgb[valid]

        if standardize:
            center = np.mean(xyz, axis=0)
            scale = np.max(xyz.max(axis=0) - xyz.min(axis=0))
            xyz = (xyz - center) / (scale + 1e-8)
            xyz[:, 2] += 0.0125

        xml = make_mitsuba_xml(
            xyz,
            rgb,
            sphere_radius=sphere_radius,
            width=width,
            height=height,
            sample_count=sample_count,
            camera_y=camera_y,
            camera_z=camera_z,
            target_z=target_z,
            fov=fov,
        )

        xml_path = os.path.join(out_dir, f"pc_render_{f:04d}.xml")
        png_path = os.path.join(out_dir, f"pc_render_{f:04d}.png")

        with open(xml_path, "w") as fp:
            fp.write(xml)

        scene = mi.load_file(xml_path)
        img = mi.render(scene)
        mi.util.write_bitmap(png_path, img)

        print(f"[✓] rendered: {png_path}")


def make_mitsuba_xml(
    xyz,
    rgb=None,
    sphere_radius=0.015,
    width=1920,
    height=1080,
    sample_count=128,
    camera_y=-3.0,
    camera_z=2.0,
    target_z=0.15,
    fov=20,
):
    head = f"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>

    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>

        <transform name="toWorld">
            <lookat origin="0.0,{camera_y},{camera_z}" target="0,0,{target_z}" up="0,0,1"/>
        </transform>

        <float name="fov" value="{fov}"/>

        <sampler type="independent">
            <integer name="sampleCount" value="{sample_count}"/>
        </sampler>

        <film type="hdrfilm">
            <integer name="width" value="{width}"/>
            <integer name="height" value="{height}"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>

    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/>
    </bsdf>
"""

    balls = []

    for i, p in enumerate(xyz):
        if rgb is not None:
            c = rgb[i].astype(np.float32)
            if c.max() > 1:
                c = c / 255.0
            r, g, b = np.clip(c, 0, 1)
        else:
            r = g = b = 1.0

        balls.append(f"""
    <shape type="sphere">
        <float name="radius" value="{sphere_radius}"/>
        <transform name="toWorld">
            <translate x="{p[0]}" y="{p[1]}" z="{p[2]}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{r},{g},{b}"/>
        </bsdf>
    </shape>
""")

    tail = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.1"/>
        </transform>
    </shape>

    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""
    return head + "".join(balls) + tail


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["export", "render", "pc_image"], required=True)

    # export paths
    parser.add_argument("--pc_zarr_path", type=str, default=None)
    parser.add_argument("--front_zarr_path", type=str, default=None)
    parser.add_argument("--wrist_zarr_path", type=str, default=None)
    parser.add_argument("--tacrgb_zarr_path", type=str, default=None)
    parser.add_argument("--tacdepth_zarr_path", type=str, default=None)
    parser.add_argument("--tacff_zarr_path", type=str, default=None)

    # output
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)

    # frame range
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)

    # rigid transform for pointcloud
    parser.add_argument("--rx", type=float, default=60)
    parser.add_argument("--ry", type=float, default=120)
    parser.add_argument("--rz", type=float, default=0)
    parser.add_argument("--tx", type=float, default=0)
    parser.add_argument("--ty", type=float, default=0)
    parser.add_argument("--tz", type=float, default=0)

    parser.add_argument("--output_png", type=str, default=None)
    parser.add_argument("--frame", type=int, default=0)

    parser.add_argument("--pc_point_size", type=float, default=40)
    parser.add_argument("--pc_elev", type=float, default=20)
    parser.add_argument("--pc_azim", type=float, default=-225)
    parser.add_argument("--pc_percentile", type=float, default=2.0)
    parser.add_argument("--pc_zoom", type=float, default=1.0)
    parser.add_argument("--pc_show_axis", action="store_true")

    # render
    parser.add_argument("--frames", type=int, nargs="+", default=[0])
    parser.add_argument("--sphere_radius", type=float, default=0.015)
    parser.add_argument("--sample_count", type=int, default=128)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera_y", type=float, default=-3.0)
    parser.add_argument("--camera_z", type=float, default=2.0)
    parser.add_argument("--target_z", type=float, default=0.15)
    parser.add_argument("--fov", type=float, default=20)

    # tacff
    parser.add_argument("--tacff_resolution", type=int, default=40)
    parser.add_argument("--tacff_arrow_scale", type=float, default=0.001)

    args = parser.parse_args()

    if args.mode == "export":
        assert args.pc_zarr_path is not None, "--pc_zarr_path is required for export"
        export_all(args)

    elif args.mode == "render":
        out_dir = args.out_dir
        if out_dir is None:
            out_dir = os.path.dirname(args.npz_path)

        render_npz_with_mitsuba(
            npz_path=args.npz_path,
            out_dir=out_dir,
            frames=args.frames,
            sphere_radius=args.sphere_radius,
            width=args.width,
            height=args.height,
            sample_count=args.sample_count,
            camera_y=args.camera_y,
            camera_z=args.camera_z,
            target_z=args.target_z,
            fov=args.fov,
        )
    elif args.mode == "pc_image":
        assert args.pc_zarr_path is not None, "--pc_zarr_path is required for pc_image"

        if args.output_png is None:
            base_dir = os.path.dirname(args.npz_path)
            args.output_png = os.path.join(base_dir, f"pointcloud_frame_{args.frame:04d}.png")

        render_single_pointcloud_image_from_zarr(
            pc_zarr_path=args.pc_zarr_path,
            output_png=args.output_png,
            frame=args.frame,
            rx=args.rx,
            ry=args.ry,
            rz=args.rz,
            elev=args.pc_elev,
            azim=args.pc_azim,
            point_size=args.pc_point_size,
            percentile=args.pc_percentile,
            zoom=args.pc_zoom,
            hide_axis=not args.pc_show_axis,
        )

        
if __name__ == "__main__":
    main()