

def escape(str):
    return str.replace("=", "\\equal").replace("|", "\\vertical")


def retrieve(str):
    return str.replace("\\equal", "=").replace("\\vertical", "|")



def createFeature(x0, xm1, y0, ym1):
    lKeyPhi = []
    if x0[0] >= "A" and x0[0] <= "Z":
        lKeyPhi.append("x_f=" + "CAPS" + "|" + "y_0=" + y0)
    elif len(x0) > 2 and "ed" == x0[-2:]:
        lKeyPhi.append("x_f=" + "SUF.ed" + "|" + "y_0=" + y0)
    elif len(x0) > 3 and "ing" == x0[-3:]:
        lKeyPhi.append("x_f=" + "SUF.ing" + "|" + "y_0=" + y0)
    elif len(x0) > 2 and "ly" == x0[-2:]:
        lKeyPhi.append("x_f=" + "SUF.ly" + "|" + "y_0=" + y0)
    elif len(x0) > 2 and "ly" == x0[-2:]:
        lKeyPhi.append("x_f=" + "SUF.er" + "|" + "y_0=" + y0)
    elif len(x0) > 3 and "ly" == x0[-3:]:
        lKeyPhi.append("x_f=" + "SUF.est" + "|" + "y_0=" + y0)
    elif len(x0) > 2 and "co" == x0[:2]:
        lKeyPhi.append("x_f=" + "PRE.co" + "|" + "y_0=" + y0)
    elif len(x0) > 3 and "non" == x0[:3]:
        lKeyPhi.append("x_f=" + "PRE.non" + "|" + "y_0=" + y0)
    elif len(x0) > 3 and "non" == x0[-3:]:
        lKeyPhi.append("x_f=" + "SUF.ion" + "|" + "y_0=" + y0)
    return lKeyPhi