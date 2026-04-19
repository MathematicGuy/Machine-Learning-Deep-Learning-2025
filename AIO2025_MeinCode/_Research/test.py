import json, os

# path = "checkpoint.json"
# tmp_path = f"{path}.tmp" # checkpoint.json.tmp

# data = {
# 	"name": "Thanh",
# 	"age": 12,
# }
# with open(path, "w", encoding="utf-8") as f:
# 	json.dump(data, f) # dump new temp data to .tmp file



# new_data = {
# 	"name": "DNT",
# 	"age": 19,
# }
# with open(tmp_path, "w", encoding="utf-8") as f:
# 	json.dump(new_data, f) # dump new temp data to .tmp file

# #? Dump data from temp file back to main file
# os.replace(tmp_path, path) # rename source "path" with the destination "tmp_path"


def read_json(file_path):
    """
    Reads JSON data from a file and returns it as a Python object.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: The loaded JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return json.loads(content)

# Example usage
data = read_json('data/wiki.json')['query']['pages'][0]
prety_json = json.dumps(data, indent=2)
print(prety_json)

"""
"content": "{
    {
        ch\u00fa th\u00edch trong b\u00e0i}}{{M\u1ed3 c\u00f4i}}<!-- *Checked -->\n\n{{DISPLAYTITLE:(202084) 2004 SE<sub>56</sub>}}\n\n'''202084 (2004 SE56)'''
        l\u00e0 m\u1ed9t [[ti\u1ec3u h\u00e0nh tinh]] [[v\u00e0nh \u0111ai ti\u1ec3u h\u00e0nh tinh|v\u00e0nh \u0111ai ch\u00ednh]] \u0111\u01b0\u1ee3c ph\u00e1t hi\u1ec7n ng\u00e0y 30 th\u00e1ng 9 n\u0103m 2004 b\u1edfi [[James Whitney Young]] \u1edf [[\u0110\u00e0i thi\u00ean v\u0103n N\u00fai b\u00e0n]] g\u1ea7n [[Wrightwood, California]].\n\n==Tham kh\u1ea3o==\n{{tham kh\u1ea3o}}\n== Li\u00ean k\u1ebft ngo\u00e0i ==\n* [http://ssd.jpl.nasa.gov/sbdb.cgi?sstr=202084 JPL Small-Body Database Browser ng\u00e0y 202084 (2004 SE56)].\n\n== Xem th\u00eam ==\n* [[Danh s\u00e1ch c\u00e1c ti\u1ec3u h\u00e0nh tinh: 202001\u2013203000]]\n\n{{MinorPlanets Navigator | | }}\n{{MinorPlanets_Footer}}\n{{Solar System}}\n\n[[Th\u1ec3 lo\u1ea1i:Ti\u1ec3u h\u00e0nh tinh v\u00e0nh \u0111ai ch\u00ednh]]\n[[Th\u1ec3 lo\u1ea1i:Thi\u00ean th\u1ec3 ph\u00e1t hi\u1ec7n n\u0103m 2004]]\n\n\n{{beltasteroid-stub}}\n\n[[en:List of minor planets: 202001\u2013203000#202001\u2013202100]]"
"""
