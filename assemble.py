import face_recognition as fr
import numpy as np
import glob
import re
import pickle

def image_to_encodings(file):
    """Return the face embeddings from an image"""
    load_image = fr.load_image_file(file)
    return fr.face_encodings(load_image)[0]

def top_match(known_encodings, names, new_image_embed):
    """With list of known image names/encodings, return closest match"""
    closests = fr.face_distance(known_encodings, new_image_embed)
    top_match = int(np.argmin(closests))
    top_matches = [int(i) for i in np.argpartition(closests, 3)[:3]]
    top_matches.remove(top_match)
    return (names[top_match], [names[i] for i in top_matches])

avengers_cast_files = glob.glob('avengers/*.jpg')
avengers_cast_names = [re.sub('.jpg|avengers/', '', image)
                            for image in avengers_cast_files]
avengers_cast_encodings = [image_to_encodings(image)
                                  for image in avengers_cast_files]
avengers_dict = {name: encodings for name, encodings in 
                            zip(avengers_cast_names, avengers_cast_encodings)}

outfile = open('avengers_dict.pickle','wb')
pickle.dump(avengers_dict,outfile)
outfile.close()