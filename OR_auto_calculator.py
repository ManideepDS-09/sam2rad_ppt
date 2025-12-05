# OR_auto_calculator.py
# ---------------------------------------------------------
# Pure OR extraction + visualization
# Inputs:
#   comps    -> list of binary component masks
#   rgb      -> original frame (RGB 1024x1024 ideally)
#   out_path -> png save path
#
# NOTE:
# DO NOT TOUCH ANY OR COMPUTATION LOGIC
# ---------------------------------------------------------

import cv2
import numpy as np

# ----------------- constants -----------------
MIN_COMP_AREA = 270


# ----------------- helpers (unchanged) -----------------
def top_y(mask01, x):
    ys = np.where(mask01[:, x] > 0)[0]
    return int(ys.min()) if len(ys) else None


def get_bounding_points(mask01):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None, None

    left_x  = int(xs.min())
    right_x = int(xs.max())

    y_left  = top_y(mask01, left_x)
    y_right = top_y(mask01, right_x)

    if y_left is None:
        y_left = int(ys.min())
    if y_right is None:
        y_right = int(ys.min())

    return (left_x, y_left), (right_x, y_right)


def compute_slope(mask01):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None

    pts_x = []
    pts_y = []

    for x in range(xs.min(), xs.max()+1):
        ys_col = np.where(mask01[:, x] > 0)[0]
        if len(ys_col) > 0:
            pts_x.append(x)
            pts_y.append(ys_col.min())

    if len(pts_x) < 2:
        return None

    pts_x = np.array(pts_x)
    pts_y = np.array(pts_y)
    A = np.vstack([pts_x, np.ones(len(pts_x))]).T
    m, c = np.linalg.lstsq(A, pts_y, rcond=None)[0]

    print("slope of middle blob =", m)
    return m


def dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def metaphyseal_zone(mask, side="left", ratio=0.1):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x_min = xs.min()
    x_max = xs.max()
    width = x_max - x_min
    w = max(1, int(width * ratio))

    if side == "left":
        zone = mask[:, x_min: x_min + w]
    else:
        zone = mask[:, x_max - w: x_max]

    return zone, x_min, x_max


# ---------------------------------------------------------
# MAIN OR EXTRACTOR
# ---------------------------------------------------------
def compute_OR_from_components(comps, rgb, out_path):

    # 1) connected comps sorted left->right
    comps_sorted = []
    for c in comps:
        ys, xs = np.where(c > 0)
        comps_sorted.append((int(xs.min()), c))

    comps_sorted.sort(key=lambda t: t[0])
    bones = [c for (_, c) in comps_sorted]

    print(f"[OR] bone count = {len(bones)}")

    # visual canvas
    vis = rgb.copy()

    # draw bone indices + bounding anchors
    for idx, comp in enumerate(bones, start=1):
        pL, pR = get_bounding_points(comp)

        if pL is not None:
            cv2.circle(vis, pL, 3, (0, 255, 255), -1)
            cv2.putText(vis, str(idx),
                        (pL[0] + 4, pL[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,255), 1, cv2.LINE_AA)

        if pR is not None:
            cv2.circle(vis, pR, 3, (0, 200, 0), -1)

        print(f"[BONE {idx}] L={pL}, R={pR}")

    # OR result vars
    OR1 = OR2 = None
    OR1_star = OR2_star = None

    h1 = H1 = h2 = H2 = None
    H1_star = H2_star = None

    # ----------------------------------------------------------------------
    # DO NOT TOUCH THE LOGIC BELOW
    # EXACT same block you wrote earlier
    # ----------------------------------------------------------------------

    # ======================= 3 bones => OR1 =======================
    if len(bones) == 3:
        A, B, C = bones

        pA_L, pA_R = get_bounding_points(A)
        pB_L, pB_R = get_bounding_points(B)
        pC_L, pC_R = get_bounding_points(C)

        if pA_R and pB_R and pC_L:
            # h1 (mm)
            h1_px = dist(pA_R, pB_R)
            h1_mm = h1_px * 0.06

            # H1 (mm)
            H1_px = dist(pA_R, pC_L)
            H1_mm = H1_px * 0.06 if H1_px > 0 else None

            OR1 = h1_mm / H1_mm if H1_mm else None
            OR1 = OR1*100

            # lines
            cv2.line(vis, pA_R, pB_R, (0,255,0), 1)
            cv2.line(vis, pA_R, pC_L, (255,0,0), 1)

            # slope part
            m = compute_slope(B)
            if m is not None:
                xA, yA = pA_R
                zoneC, x_min_C, _ = metaphyseal_zone(C, side="left", ratio=0.12)
                if zoneC is not None:
                    C_meta = zoneC.astype(bool)
                    H_zone, W_zone = C_meta.shape
                    hit = None

                    for shift in range(0,80):
                        k = (yA + shift) - m*xA

                        for step in range(1500):
                            xg = xA + step
                            if xg >= x_min_C + W_zone:
                                break

                            xl = xg - x_min_C
                            yg = int(m*xg + k)

                            if 0<=xl<W_zone and 0<=yg<H_zone and C_meta[yg,xl]:
                                hit = (xg,yg)
                                break
                        if hit: break

                    if hit:
                        H1_star_px = dist(pA_R, hit)
                        H1_star_mm = H1_star_px * 0.06
                        if H1_star_mm>0:
                            OR1_star = (h1_mm/H1_star_mm)*100
                        cv2.line(vis,pA_R,hit,(0,128,255),1)

            # assign outputs
            h1 = h1_mm
            H1 = H1_mm
            H1_star = H1_star_mm if "H1_star_mm" in locals() else None

    # ======================= 4 bones => OR1 + OR2 =======================
    elif len(bones) == 4:
        # EXACT original block preserved
        A, B, C, D = bones

        pA_L, pA_R = get_bounding_points(A)
        pB_L, pB_R = get_bounding_points(B)
        pC_L, pC_R = get_bounding_points(C)
        pD_L, pD_R = get_bounding_points(D)

        # h1
        if pA_R and pB_R:
            h1_px = dist(pA_R,pB_R)
            h1 = h1_px * 0.06

        # H1
        if pA_R and pC_L:
            H1_px = dist(pA_R,pC_L)
            H1 = H1_px * 0.06

        if h1 and H1 and H1>0:
            OR1 = (h1/H1)*100

        # h2
        if pB_R and pC_R:
            h2_px = dist(pB_R,pC_R)
            h2 = h2_px * 0.06

        # H2
        if pB_R and pD_L:
            H2_px = dist(pB_R,pD_L)
            H2 = H2_px * 0.06

        if h2 and H2 and H2>0:
            OR2 = (h2/H2)*100

        # debug lines
        if pA_R and pB_R:
            cv2.line(vis,pA_R,pB_R,(255,255,0),1)
        if pA_R and pC_L:
            cv2.line(vis,pA_R,pC_L,(255,255,0),1)
        if pB_R and pC_R:
            cv2.line(vis,pB_R,pC_R,(255,255,255),1)
        if pB_R and pD_L:
            cv2.line(vis,pB_R,pD_L,(255,255,255),1)

        # slope OR1*
        m1 = compute_slope(B)
        if m1 and pA_R:
            zone,x_min_C,_ = metaphyseal_zone(C,"left",0.12)
            if zone is not None:
                C_meta = zone.astype(bool)
                xA,yA = pA_R
                H_zone,W_zone = C_meta.shape
                hit=None

                for shift in range(0,80):
                    k1=(yA+shift)-m1*xA

                    for step in range(1500):
                        xg = xA + step
                        if xg>=x_min_C+W_zone:
                            break
                        xl = xg-x_min_C
                        yg = int(m1*xg + k1)
                        if 0<=xl<W_zone and 0<=yg<H_zone and C_meta[yg,xl]:
                            hit=(xg,yg);break
                    if hit:break

                if hit:
                    H1_star_px = dist(pA_R,hit)
                    H1_star = H1_star_px*0.06
                    if H1_star>0:
                        OR1_star = (h1/H1_star)*100
                    cv2.line(vis,pA_R,hit,(0,128,255),1)

        # slope OR2*
        m2 = compute_slope(C)
        if m2 and pB_R:
            zone,x_min_D,_ = metaphyseal_zone(D,"left",0.12)
            if zone is not None:
                D_meta=zone.astype(bool)
                xB,yB=pB_R
                H_zone,W_zone=D_meta.shape
                hit=None

                for shift in range(0,80):
                    k2=(yB+shift)-m2*xB

                    for step in range(1500):
                        xg=xB+step
                        if xg>=x_min_D+W_zone:
                            break
                        xl=xg-x_min_D
                        yg=int(m2*xg+k2)
                        if 0<=xl<W_zone and 0<=yg<H_zone and D_meta[yg,xl]:
                            hit=(xg,yg);break
                    if hit: break

                if hit:
                    H2_star_px=dist(pB_R,hit)
                    H2_star=H2_star_px*0.06
                    if H2_star>0:
                        OR2_star=(h2/H2_star)*100
                    cv2.line(vis,pB_R,hit,(0,128,255),1)

    # ======================= >=5 bones => OR1 + OR2 =======================
    elif len(bones) >= 5:
        # EXACT original logic preserved
        A,B,C,D,E=bones[:5]

        pA_L,pA_R=get_bounding_points(A)
        pB_L,pB_R=get_bounding_points(B)
        pC_L,pC_R=get_bounding_points(C)
        pD_L,pD_R=get_bounding_points(D)
        pE_L,pE_R=get_bounding_points(E)

        if pA_R and pB_R:
            h1=dist(pA_R,pB_R)*0.06
        if pA_R and pC_L:
            H1=dist(pA_R,pC_L)*0.06
        if h1 and H1 and H1>0:
            OR1=((h1*0.06)/(H1*0.06))*100

        if pC_R and pD_R:
            h2=dist(pC_R,pD_R)*0.06
        if pC_R and pE_L:
            H2=dist(pC_R,pE_L)*0.06
        if h2 and H2 and H2>0:
            OR2=((h2*0.06)/(H2*0.06))*100

        # lines
        if h1: cv2.line(vis,pA_R,pB_R,(255,255,0),1)
        if H1: cv2.line(vis,pA_R,pC_L,(255,255,0),1)
        if h2: cv2.line(vis,pC_R,pD_R,(255,255,255),1)
        if H2: cv2.line(vis,pC_R,pE_L,(255,255,255),1)

        # slope OR1*
        m1=compute_slope(B)
        if m1 and pA_R:
            zone,xmin,_=metaphyseal_zone(C,"left",0.12)
            if zone is not None:
                C_meta=zone.astype(bool)
                xA,yA=pA_R
                H_zone,W_zone=C_meta.shape
                hit=None

                for shift in range(0,80):
                    k1=(yA+shift)-m1*xA
                    for step in range(1500):
                        xg=xA+step
                        if xg>=xmin+W_zone:
                            break
                        xl=xg-xmin
                        yg=int(m1*xg+k1)
                        if 0<=xl<W_zone and 0<=yg<H_zone and C_meta[yg,xl]:
                            hit=(xg,yg);break
                    if hit:break

                if hit:
                    H1_star=dist(pA_R,hit)*0.06
                    if H1_star>0:
                        OR1_star=((h1*0.06)/(H1_star*0.06))*100
                    cv2.line(vis,pA_R,hit,(0,128,255),1)

        # slope OR2*
        m2=compute_slope(D)
        if m2 and pC_R:
            zone,xmin,_=metaphyseal_zone(E,"left",0.12)
            if zone is not None:
                E_meta=zone.astype(bool)
                xC,yC=pC_R
                H_zone,W_zone=E_meta.shape
                hit=None

                for shift in range(0,80):
                    k2=(yC+shift)-m2*xC
                    for step in range(1500):
                        xg=xC+step
                        if xg>=xmin+W_zone:
                            break
                        xl=xg-xmin
                        yg=int(m2*xg+k2)
                        if 0<=xl<W_zone and 0<=yg<H_zone and E_meta[yg,xl]:
                            hit=(xg,yg);break
                    if hit:break

                if hit:
                    H2_star=dist(pC_R,hit)*0.06
                    if H2_star>0:
                        OR2_star=((h2*0.06)/(H2_star*0.06))*100
                    cv2.line(vis,pC_R,hit,(0,128,255),1)

    # ===================================================
    # irregular bone count
    # ===================================================
    else:
        print(f"[OR] irregular bone count {len(bones)}")

    # ---------------- overlay text -----------------
    ytxt = 20
    if OR1 is not None:
        cv2.putText(vis,f"OR1={OR1*100:.1f}%",(10,ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),1)
    ytxt+=20
    if OR2 is not None:
        cv2.putText(vis,f"OR2={OR2*100:.1f}%",(10,ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),1)
    ytxt+=20
    if OR1_star is not None:
        cv2.putText(vis,f"OR1*={OR1_star*100:.1f}%",(10,ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,128,255),1)
    ytxt+=20
    if OR2_star is not None:
        cv2.putText(vis,f"OR2*={OR2_star*100:.1f}%",(10,ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,128,255),1)

    # only save if a valid out_path is provided
    if out_path:
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        if not any(out_path.lower().endswith(e) for e in valid_exts):
            out_path += ".png"
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print("[OR] debug saved:", out_path)

    return {
        "bone_count": len(bones),
        "h1": h1,   
        "H1": H1,
        "OR1 (%)": OR1,
        "H1_*": H1_star,
        "OR1_* (%)": OR1_star,
        
        "h2": h2,
        "H2": H2,      
        "OR2 (%)" : OR2,
        "H2_*": H2_star,
        "OR2_* (%)": OR2_star,
        
        
    }