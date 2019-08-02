

def escape(str):
    return str.replace("=", "\\equal").replace("|", "\\vertical")


def retrieve(str):
    return str.replace("\\equal", "=").replace("\\vertical", "|")



def createFeature(x0, xm1, y0, ym1):
    lKeyPhi = ["<bias_xxxx>"]
    lKeyPhi.append("x_0=" + x0 + "|" + "y_0=" + y0)
    lKeyPhi.append("y_0=" + y0 + "|" + "y_-1=" + ym1)
    if x0[0] >= "A" and x0[0] <= "Z":
        lKeyPhi.append("x_f=" + "CAPS" + "|" + "y_0=" + y0)

    if len(x0) > 2: 
        if "ed" == x0[-2:]:
            lKeyPhi.append("x_f=" + "SUF.ed" + "|" + "y_0=" + y0)
        if "ly" == x0[-2:]:
            lKeyPhi.append("x_f=" + "SUF.ly" + "|" + "y_0=" + y0)
        if "co" == x0[:2]:
            lKeyPhi.append("x_f=" + "PRE.co" + "|" + "y_0=" + y0)
        if "un" == x0[:2]:
            lKeyPhi.append("x_f=" + "PRE.un" + "|" + "y_0=" + y0)
        if "re" == x0[:2]:
            lKeyPhi.append("x_f=" + "PRE.re" + "|" + "y_0=" + y0)
        if "im" == x0[:2]:
            lKeyPhi.append("x_f=" + "PRE.im" + "|" + "y_0=" + y0)
        if "il" == x0[:2]:
            lKeyPhi.append("x_f=" + "PRE.il" + "|" + "y_0=" + y0)

    if len(x0) > 3:
        if "ing" == x0[-3:]:
            lKeyPhi.append("x_f=" + "SUF.ing" + "|" + "y_0=" + y0)
        if "pre" == x0[-3:]:
            lKeyPhi.append("x_f=" + "PRE.pre" + "|" + "y_0=" + y0)
        if "pro" == x0[-3:]:
            lKeyPhi.append("x_f=" + "PRE.pro" + "|" + "y_0=" + y0)
        if "non" == x0[:3]:
            lKeyPhi.append("x_f=" + "PRE.non" + "|" + "y_0=" + y0)
        if "mis" == x0[:3]:
            lKeyPhi.append("x_f=" + "PRE.mis" + "|" + "y_0=" + y0)
        if "dis" == x0[:3]:
            lKeyPhi.append("x_f=" + "PRE.dis" + "|" + "y_0=" + y0)
    if "take" in x0:
        lKeyPhi.append("x_f=" + "IN.take" + "|" + "y_0=" + y0)

    

    return lKeyPhi