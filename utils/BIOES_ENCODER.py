

# TAG_TO_NUNBER = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'E-ORG': 3, 'S-ORG': 4, 'B-MISC': 5, 'I-MISC': 6, 'E-MISC': 7, 'S-MISC': 8, 'B-PER': 9, 'I-PER': 10, 'E-PER': 11, 'S-PER': 12, 'B-LOC': 13, 'I-LOC': 14, 'E-LOC': 15, 'S-LOC': 16, 'B-WORK_OF_ART': 17, 'I-WORK_OF_ART': 18, 'E-WORK_OF_ART': 19, 'S-WORK_OF_ART': 20, 'B-CARDINAL': 21, 'I-CARDINAL': 22, 'E-CARDINAL': 23, 'S-CARDINAL': 24, 'B-EVENT': 25, 'I-EVENT': 26, 'E-EVENT': 27, 'S-EVENT': 28, 'B-NORP': 29, 'I-NORP': 30, 'E-NORP': 31, 'S-NORP': 32, 'B-GPE': 33, 'I-GPE': 34, 'E-GPE': 35, 'S-GPE': 36, 'B-DATE': 37, 'I-DATE': 38, 'E-DATE': 39, 'S-DATE': 40, 'B-FAC': 41, 'I-FAC': 42, 'E-FAC': 43, 'S-FAC': 44, 'B-QUANTITY': 45, 'I-QUANTITY': 46, 'E-QUANTITY': 47, 'S-QUANTITY': 48, 'B-ORDINAL': 49, 'I-ORDINAL': 50, 'E-ORDINAL': 51, 'S-ORDINAL': 52, 'B-TIME': 53, 'I-TIME': 54, 'E-TIME': 55, 'S-TIME': 56, 'B-PRODUCT': 57, 'I-PRODUCT': 58, 'E-PRODUCT': 59, 'S-PRODUCT': 60, 'B-PERCENT': 61, 'I-PERCENT': 62, 'E-PERCENT': 63, 'S-PERCENT': 64, 'B-MONEY': 65, 'I-MONEY': 66, 'E-MONEY': 67, 'S-MONEY': 68, 'B-LAW': 69, 'I-LAW': 70, 'E-LAW': 71, 'S-LAW': 72, 'B-LANGUAGE': 73, 'I-LANGUAGE': 74, 'E-LANGUAGE': 75, 'S-LANGUAGE': 76, 'B-group': 77, 'I-group': 78, 'E-group': 79, 'S-group': 80, 'B-corporation': 81, 'I-corporation': 82, 'E-corporation': 83, 'S-corporation': 84, 'B-creative-work': 85, 'I-creative-work': 86, 'E-creative-work': 87, 'S-creative-work': 88, 'B-product': 89, 'I-product': 90, 'E-product': 91, 'S-product': 92, 'B-Gene_or_gene_product': 93, 'I-Gene_or_gene_product': 94, 'E-Gene_or_gene_product': 95, 'S-Gene_or_gene_product': 96, 'B-Complex': 97, 'I-Complex': 98, 'E-Complex': 99, 'S-Complex': 100, 'B-Cellular_component': 101, 'I-Cellular_component': 102, 'E-Cellular_component': 103, 'S-Cellular_component': 104, 'B-Simple_chemical': 105, 'I-Simple_chemical': 106, 'E-Simple_chemical': 107, 'S-Simple_chemical': 108, 'B-Temporal': 109, 'I-Temporal': 110, 'E-Temporal': 111, 'S-Temporal': 112, 'B-DocumentReference': 113, 'I-DocumentReference': 114, 'E-DocumentReference': 115, 'S-DocumentReference': 116, 'B-Quantity': 117, 'I-Quantity': 118, 'E-Quantity': 119, 'S-Quantity': 120, 'B-Nationality': 121, 'I-Nationality': 122, 'E-Nationality': 123, 'S-Nationality': 124, 'B-MilitaryPlatform': 125, 'I-MilitaryPlatform': 126, 'E-MilitaryPlatform': 127, 'S-MilitaryPlatform': 128, 'B-Weapon': 129, 'I-Weapon': 130, 'E-Weapon': 131, 'S-Weapon': 132, 'B-Money': 133, 'I-Money': 134, 'E-Money': 135, 'S-Money': 136, 'B-CHARACTER': 137, 'I-CHARACTER': 138, 'E-CHARACTER': 139, 'S-CHARACTER': 140, 'B-YEAR': 141, 'I-YEAR': 142, 'E-YEAR': 143, 'S-YEAR': 144, 'B-TITLE': 145, 'I-TITLE': 146, 'E-TITLE': 147, 'S-TITLE': 148, 'B-SONG': 149, 'I-SONG': 150, 'E-SONG': 151, 'S-SONG': 152, 'B-ACTOR': 153, 'I-ACTOR': 154, 'E-ACTOR': 155, 'S-ACTOR': 156, 'B-PLOT': 157, 'I-PLOT': 158, 'E-PLOT': 159, 'S-PLOT': 160, 'B-GENRE': 161, 'I-GENRE': 162, 'E-GENRE': 163, 'S-GENRE': 164, 'B-RATING': 165, 'I-RATING': 166, 'E-RATING': 167, 'S-RATING': 168, 'B-DIRECTOR': 169, 'I-DIRECTOR': 170, 'E-DIRECTOR': 171, 'S-DIRECTOR': 172, 'B-RATINGS_AVERAGE': 173, 'I-RATINGS_AVERAGE': 174, 'E-RATINGS_AVERAGE': 175, 'S-RATINGS_AVERAGE': 176, 'B-TRAILER': 177, 'I-TRAILER': 178, 'E-TRAILER': 179, 'S-TRAILER': 180, 'B-REVIEW': 181, 'I-REVIEW': 182, 'E-REVIEW': 183, 'S-REVIEW': 184, 'B-Restaurant_Name': 185, 'I-Restaurant_Name': 186, 'E-Restaurant_Name': 187, 'S-Restaurant_Name': 188, 'B-Rating': 189, 'I-Rating': 190, 'E-Rating': 191, 'S-Rating': 192, 'B-Hours': 193, 'I-Hours': 194, 'E-Hours': 195, 'S-Hours': 196, 'B-Dish': 197, 'I-Dish': 198, 'E-Dish': 199, 'S-Dish': 200, 'B-Cuisine': 201, 'I-Cuisine': 202, 'E-Cuisine': 203, 'S-Cuisine': 204, 'B-Amenity': 205, 'I-Amenity': 206, 'E-Amenity': 207, 'S-Amenity': 208, 'B-Price': 209, 'I-Price': 210, 'E-Price': 211, 'S-Price': 212}
# # INDEX_TO_TYPE = {1: 0, 4: 0, 5: 1, 8: 1, 9: 2, 12: 2, 13: 3, 16: 3, 17: 4, 20: 4, 21: 5, 24: 5, 25: 6, 28: 6, 29: 7, 32: 7, 33: 8, 36: 8, 37: 9, 40: 9, 41: 10, 44: 10, 45: 11, 48: 11, 49: 12, 52: 12, 53: 13, 56: 13, 57: 14, 60: 14, 61: 15, 64: 15, 65: 16, 68: 16, 69: 17, 72: 17, 73: 18, 76: 18, 77: 19, 80: 19, 81: 20, 84: 20, 85: 21, 88: 21, 89: 22, 92: 22, 93: 23, 96: 23, 97: 24, 100: 24, 101: 25, 104: 25, 105: 26, 108: 26, 109: 27, 112: 27, 113: 28, 116: 28, 117: 29, 120: 29, 121: 30, 124: 30, 125: 31, 128: 31, 129: 32, 132: 32, 133: 33, 136: 33, 137: 34, 140: 34, 141: 35, 144: 35, 145: 36, 148: 36, 149: 37, 152: 37, 153: 38, 156: 38, 157: 39, 160: 39, 161: 40, 164: 40, 165: 41, 168: 41, 169: 42, 172: 42, 173: 43, 176: 43, 177: 44, 180: 44, 181: 45, 184: 45, 185: 46, 188: 46, 189: 47, 192: 47, 193: 48, 196: 48, 197: 49, 200: 49, 201: 50, 204: 50, 205: 51, 208: 51, 209: 52, 212: 52}
# #
# # B_INDEX = {1: 'B-ORG', 5: 'B-MISC', 9: 'B-PER', 13: 'B-LOC', 17: 'B-WORK_OF_ART', 21: 'B-CARDINAL', 25: 'B-EVENT', 29: 'B-NORP', 33: 'B-GPE', 37: 'B-DATE', 41: 'B-FAC', 45: 'B-QUANTITY', 49: 'B-ORDINAL', 53: 'B-TIME', 57: 'B-PRODUCT', 61: 'B-PERCENT', 65: 'B-MONEY', 69: 'B-LAW', 73: 'B-LANGUAGE', 77: 'B-group', 81: 'B-corporation', 85: 'B-creative-work', 89: 'B-product', 93: 'B-Gene_or_gene_product', 97: 'B-Complex', 101: 'B-Cellular_component', 105: 'B-Simple_chemical', 109: 'B-Temporal', 113: 'B-DocumentReference', 117: 'B-Quantity', 121: 'B-Nationality', 125: 'B-MilitaryPlatform', 129: 'B-Weapon', 133: 'B-Money', 137: 'B-CHARACTER', 141: 'B-YEAR', 145: 'B-TITLE', 149: 'B-SONG', 153: 'B-ACTOR', 157: 'B-PLOT', 161: 'B-GENRE', 165: 'B-RATING', 169: 'B-DIRECTOR', 173: 'B-RATINGS_AVERAGE', 177: 'B-TRAILER', 181: 'B-REVIEW', 185: 'B-Restaurant_Name', 189: 'B-Rating', 193: 'B-Hours', 197: 'B-Dish', 201: 'B-Cuisine', 205: 'B-Amenity', 209: 'B-Price'}
# # E_INDEX = {3: 'E-ORG', 7: 'E-MISC', 11: 'E-PER', 15: 'E-LOC', 19: 'E-WORK_OF_ART', 23: 'E-CARDINAL', 27: 'E-EVENT', 31: 'E-NORP', 35: 'E-GPE', 39: 'E-DATE', 43: 'E-FAC', 47: 'E-QUANTITY', 51: 'E-ORDINAL', 55: 'E-TIME', 59: 'E-PRODUCT', 63: 'E-PERCENT', 67: 'E-MONEY', 71: 'E-LAW', 75: 'E-LANGUAGE', 79: 'E-group', 83: 'E-corporation', 87: 'E-creative-work', 91: 'E-product', 95: 'E-Gene_or_gene_product', 99: 'E-Complex', 103: 'E-Cellular_component', 107: 'E-Simple_chemical', 111: 'E-Temporal', 115: 'E-DocumentReference', 119: 'E-Quantity', 123: 'E-Nationality', 127: 'E-MilitaryPlatform', 131: 'E-Weapon', 135: 'E-Money', 139: 'E-CHARACTER', 143: 'E-YEAR', 147: 'E-TITLE', 151: 'E-SONG', 155: 'E-ACTOR', 159: 'E-PLOT', 163: 'E-GENRE', 167: 'E-RATING', 171: 'E-DIRECTOR', 175: 'E-RATINGS_AVERAGE', 179: 'E-TRAILER', 183: 'E-REVIEW', 187: 'E-Restaurant_Name', 191: 'E-Rating', 195: 'E-Hours', 199: 'E-Dish', 203: 'E-Cuisine', 207: 'E-Amenity', 211: 'E-Price'}
# # S_INDEX = {4: 'S-ORG', 8: 'S-MISC', 12: 'S-PER', 16: 'S-LOC', 20: 'S-WORK_OF_ART', 24: 'S-CARDINAL', 28: 'S-EVENT', 32: 'S-NORP', 36: 'S-GPE', 40: 'S-DATE', 44: 'S-FAC', 48: 'S-QUANTITY', 52: 'S-ORDINAL', 56: 'S-TIME', 60: 'S-PRODUCT', 64: 'S-PERCENT', 68: 'S-MONEY', 72: 'S-LAW', 76: 'S-LANGUAGE', 80: 'S-group', 84: 'S-corporation', 88: 'S-creative-work', 92: 'S-product', 96: 'S-Gene_or_gene_product', 100: 'S-Complex', 104: 'S-Cellular_component', 108: 'S-Simple_chemical', 112: 'S-Temporal', 116: 'S-DocumentReference', 120: 'S-Quantity', 124: 'S-Nationality', 128: 'S-MilitaryPlatform', 132: 'S-Weapon', 136: 'S-Money', 140: 'S-CHARACTER', 144: 'S-YEAR', 148: 'S-TITLE', 152: 'S-SONG', 156: 'S-ACTOR', 160: 'S-PLOT', 164: 'S-GENRE', 168: 'S-RATING', 172: 'S-DIRECTOR', 176: 'S-RATINGS_AVERAGE', 180: 'S-TRAILER', 184: 'S-REVIEW', 188: 'S-Restaurant_Name', 192: 'S-Rating', 196: 'S-Hours', 200: 'S-Dish', 204: 'S-Cuisine', 208: 'S-Amenity', 212: 'S-Price'}
# RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20|21[,22]*,23|24|25[,26]*,27|28|29[,30]*,31|32|33[,34]*,35|36|37[,38]*,39|40|41[,42]*,43|44|45[,46]*,47|48|49[,50]*,51|52|53[,54]*,55|56|57[,58]*,59|60|61[,62]*,63|64|65[,66]*,67|68|69[,70]*,71|72|73[,74]*,75|76|77[,78]*,79|80|81[,82]*,83|84|85[,86]*,87|88|89[,90]*,91|92|93[,94]*,95|96|97[,98]*,99|100|101[,102]*,103|104|105[,106]*,107|108|109[,110]*,111|112|113[,114]*,115|116|117[,118]*,119|120|121[,122]*,123|124|125[,126]*,127|128|129[,130]*,131|132|133[,134]*,135|136|137[,138]*,139|140|141[,142]*,143|144|145[,146]*,147|148|149[,150]*,151|152|153[,154]*,155|156|157[,158]*,159|160|161[,162]*,163|164|165[,166]*,167|168|169[,170]*,171|172|173[,174]*,175|176|177[,178]*,179|180|181[,182]*,183|184|185[,186]*,187|188|189[,190]*,191|192|193[,194]*,195|196|197[,198]*,199|200|201[,202]*,203|204|205[,206]*,207|208|209[,210]*,211|212'
#


def get_TAG_and_pattens(model):
    if model == 'combine' or model == 'single_combine':
        TAG_TO_NUNBER = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'E-ORG': 3, 'S-ORG': 4, 'B-MISC': 5, 'I-MISC': 6, 'E-MISC': 7,
                         'S-MISC': 8, 'B-PER': 9, 'I-PER': 10, 'E-PER': 11, 'S-PER': 12, 'B-LOC': 13, 'I-LOC': 14,
                         'E-LOC': 15, 'S-LOC': 16, 'B-WORK_OF_ART': 17, 'I-WORK_OF_ART': 18, 'E-WORK_OF_ART': 19,
                         'S-WORK_OF_ART': 20, 'B-CARDINAL': 21, 'I-CARDINAL': 22, 'E-CARDINAL': 23, 'S-CARDINAL': 24,
                         'B-EVENT': 25, 'I-EVENT': 26, 'E-EVENT': 27, 'S-EVENT': 28, 'B-NORP': 29, 'I-NORP': 30,
                         'E-NORP': 31, 'S-NORP': 32, 'B-GPE': 33, 'I-GPE': 34, 'E-GPE': 35, 'S-GPE': 36, 'B-DATE': 37,
                         'I-DATE': 38, 'E-DATE': 39, 'S-DATE': 40, 'B-FAC': 41, 'I-FAC': 42, 'E-FAC': 43, 'S-FAC': 44,
                         'B-QUANTITY': 45, 'I-QUANTITY': 46, 'E-QUANTITY': 47, 'S-QUANTITY': 48, 'B-ORDINAL': 49,
                         'I-ORDINAL': 50, 'E-ORDINAL': 51, 'S-ORDINAL': 52, 'B-TIME': 53, 'I-TIME': 54, 'E-TIME': 55,
                         'S-TIME': 56, 'B-PRODUCT': 57, 'I-PRODUCT': 58, 'E-PRODUCT': 59, 'S-PRODUCT': 60,
                         'B-PERCENT': 61, 'I-PERCENT': 62, 'E-PERCENT': 63, 'S-PERCENT': 64, 'B-MONEY': 65,
                         'I-MONEY': 66, 'E-MONEY': 67, 'S-MONEY': 68, 'B-LAW': 69, 'I-LAW': 70, 'E-LAW': 71,
                         'S-LAW': 72, 'B-LANGUAGE': 73, 'I-LANGUAGE': 74, 'E-LANGUAGE': 75, 'S-LANGUAGE': 76,
                         'B-group': 77, 'I-group': 78, 'E-group': 79, 'S-group': 80, 'B-corporation': 81,
                         'I-corporation': 82, 'E-corporation': 83, 'S-corporation': 84, 'B-creative-work': 85,
                         'I-creative-work': 86, 'E-creative-work': 87, 'S-creative-work': 88, 'B-product': 89,
                         'I-product': 90, 'E-product': 91, 'S-product': 92, 'B-Gene_or_gene_product': 93,
                         'I-Gene_or_gene_product': 94, 'E-Gene_or_gene_product': 95, 'S-Gene_or_gene_product': 96,
                         'B-Complex': 97, 'I-Complex': 98, 'E-Complex': 99, 'S-Complex': 100,
                         'B-Cellular_component': 101, 'I-Cellular_component': 102, 'E-Cellular_component': 103,
                         'S-Cellular_component': 104, 'B-Simple_chemical': 105, 'I-Simple_chemical': 106,
                         'E-Simple_chemical': 107, 'S-Simple_chemical': 108, 'B-Temporal': 109, 'I-Temporal': 110,
                         'E-Temporal': 111, 'S-Temporal': 112, 'B-DocumentReference': 113, 'I-DocumentReference': 114,
                         'E-DocumentReference': 115, 'S-DocumentReference': 116, 'B-Quantity': 117, 'I-Quantity': 118,
                         'E-Quantity': 119, 'S-Quantity': 120, 'B-Nationality': 121, 'I-Nationality': 122,
                         'E-Nationality': 123, 'S-Nationality': 124, 'B-MilitaryPlatform': 125,
                         'I-MilitaryPlatform': 126, 'E-MilitaryPlatform': 127, 'S-MilitaryPlatform': 128,
                         'B-Weapon': 129, 'I-Weapon': 130, 'E-Weapon': 131, 'S-Weapon': 132, 'B-Money': 133,
                         'I-Money': 134, 'E-Money': 135, 'S-Money': 136, 'B-CHARACTER': 137, 'I-CHARACTER': 138,
                         'E-CHARACTER': 139, 'S-CHARACTER': 140, 'B-YEAR': 141, 'I-YEAR': 142, 'E-YEAR': 143,
                         'S-YEAR': 144, 'B-TITLE': 145, 'I-TITLE': 146, 'E-TITLE': 147, 'S-TITLE': 148, 'B-SONG': 149,
                         'I-SONG': 150, 'E-SONG': 151, 'S-SONG': 152, 'B-ACTOR': 153, 'I-ACTOR': 154, 'E-ACTOR': 155,
                         'S-ACTOR': 156, 'B-PLOT': 157, 'I-PLOT': 158, 'E-PLOT': 159, 'S-PLOT': 160, 'B-GENRE': 161,
                         'I-GENRE': 162, 'E-GENRE': 163, 'S-GENRE': 164, 'B-RATING': 165, 'I-RATING': 166,
                         'E-RATING': 167, 'S-RATING': 168, 'B-DIRECTOR': 169, 'I-DIRECTOR': 170, 'E-DIRECTOR': 171,
                         'S-DIRECTOR': 172, 'B-RATINGS_AVERAGE': 173, 'I-RATINGS_AVERAGE': 174,
                         'E-RATINGS_AVERAGE': 175, 'S-RATINGS_AVERAGE': 176, 'B-TRAILER': 177, 'I-TRAILER': 178,
                         'E-TRAILER': 179, 'S-TRAILER': 180, 'B-REVIEW': 181, 'I-REVIEW': 182, 'E-REVIEW': 183,
                         'S-REVIEW': 184, 'B-Restaurant_Name': 185, 'I-Restaurant_Name': 186, 'E-Restaurant_Name': 187,
                         'S-Restaurant_Name': 188, 'B-Rating': 189, 'I-Rating': 190, 'E-Rating': 191, 'S-Rating': 192,
                         'B-Hours': 193, 'I-Hours': 194, 'E-Hours': 195, 'S-Hours': 196, 'B-Dish': 197, 'I-Dish': 198,
                         'E-Dish': 199, 'S-Dish': 200, 'B-Cuisine': 201, 'I-Cuisine': 202, 'E-Cuisine': 203,
                         'S-Cuisine': 204, 'B-Amenity': 205, 'I-Amenity': 206, 'E-Amenity': 207, 'S-Amenity': 208,
                         'B-Price': 209, 'I-Price': 210, 'E-Price': 211, 'S-Price': 212}
        # INDEX_TO_TYPE = {1: 0, 4: 0, 5: 1, 8: 1, 9: 2, 12: 2, 13: 3, 16: 3, 17: 4, 20: 4, 21: 5, 24: 5, 25: 6, 28: 6, 29: 7, 32: 7, 33: 8, 36: 8, 37: 9, 40: 9, 41: 10, 44: 10, 45: 11, 48: 11, 49: 12, 52: 12, 53: 13, 56: 13, 57: 14, 60: 14, 61: 15, 64: 15, 65: 16, 68: 16, 69: 17, 72: 17, 73: 18, 76: 18, 77: 19, 80: 19, 81: 20, 84: 20, 85: 21, 88: 21, 89: 22, 92: 22, 93: 23, 96: 23, 97: 24, 100: 24, 101: 25, 104: 25, 105: 26, 108: 26, 109: 27, 112: 27, 113: 28, 116: 28, 117: 29, 120: 29, 121: 30, 124: 30, 125: 31, 128: 31, 129: 32, 132: 32, 133: 33, 136: 33, 137: 34, 140: 34, 141: 35, 144: 35, 145: 36, 148: 36, 149: 37, 152: 37, 153: 38, 156: 38, 157: 39, 160: 39, 161: 40, 164: 40, 165: 41, 168: 41, 169: 42, 172: 42, 173: 43, 176: 43, 177: 44, 180: 44, 181: 45, 184: 45, 185: 46, 188: 46, 189: 47, 192: 47, 193: 48, 196: 48, 197: 49, 200: 49, 201: 50, 204: 50, 205: 51, 208: 51, 209: 52, 212: 52}
        #
        # B_INDEX = {1: 'B-ORG', 5: 'B-MISC', 9: 'B-PER', 13: 'B-LOC', 17: 'B-WORK_OF_ART', 21: 'B-CARDINAL', 25: 'B-EVENT', 29: 'B-NORP', 33: 'B-GPE', 37: 'B-DATE', 41: 'B-FAC', 45: 'B-QUANTITY', 49: 'B-ORDINAL', 53: 'B-TIME', 57: 'B-PRODUCT', 61: 'B-PERCENT', 65: 'B-MONEY', 69: 'B-LAW', 73: 'B-LANGUAGE', 77: 'B-group', 81: 'B-corporation', 85: 'B-creative-work', 89: 'B-product', 93: 'B-Gene_or_gene_product', 97: 'B-Complex', 101: 'B-Cellular_component', 105: 'B-Simple_chemical', 109: 'B-Temporal', 113: 'B-DocumentReference', 117: 'B-Quantity', 121: 'B-Nationality', 125: 'B-MilitaryPlatform', 129: 'B-Weapon', 133: 'B-Money', 137: 'B-CHARACTER', 141: 'B-YEAR', 145: 'B-TITLE', 149: 'B-SONG', 153: 'B-ACTOR', 157: 'B-PLOT', 161: 'B-GENRE', 165: 'B-RATING', 169: 'B-DIRECTOR', 173: 'B-RATINGS_AVERAGE', 177: 'B-TRAILER', 181: 'B-REVIEW', 185: 'B-Restaurant_Name', 189: 'B-Rating', 193: 'B-Hours', 197: 'B-Dish', 201: 'B-Cuisine', 205: 'B-Amenity', 209: 'B-Price'}
        # E_INDEX = {3: 'E-ORG', 7: 'E-MISC', 11: 'E-PER', 15: 'E-LOC', 19: 'E-WORK_OF_ART', 23: 'E-CARDINAL', 27: 'E-EVENT', 31: 'E-NORP', 35: 'E-GPE', 39: 'E-DATE', 43: 'E-FAC', 47: 'E-QUANTITY', 51: 'E-ORDINAL', 55: 'E-TIME', 59: 'E-PRODUCT', 63: 'E-PERCENT', 67: 'E-MONEY', 71: 'E-LAW', 75: 'E-LANGUAGE', 79: 'E-group', 83: 'E-corporation', 87: 'E-creative-work', 91: 'E-product', 95: 'E-Gene_or_gene_product', 99: 'E-Complex', 103: 'E-Cellular_component', 107: 'E-Simple_chemical', 111: 'E-Temporal', 115: 'E-DocumentReference', 119: 'E-Quantity', 123: 'E-Nationality', 127: 'E-MilitaryPlatform', 131: 'E-Weapon', 135: 'E-Money', 139: 'E-CHARACTER', 143: 'E-YEAR', 147: 'E-TITLE', 151: 'E-SONG', 155: 'E-ACTOR', 159: 'E-PLOT', 163: 'E-GENRE', 167: 'E-RATING', 171: 'E-DIRECTOR', 175: 'E-RATINGS_AVERAGE', 179: 'E-TRAILER', 183: 'E-REVIEW', 187: 'E-Restaurant_Name', 191: 'E-Rating', 195: 'E-Hours', 199: 'E-Dish', 203: 'E-Cuisine', 207: 'E-Amenity', 211: 'E-Price'}
        # S_INDEX = {4: 'S-ORG', 8: 'S-MISC', 12: 'S-PER', 16: 'S-LOC', 20: 'S-WORK_OF_ART', 24: 'S-CARDINAL', 28: 'S-EVENT', 32: 'S-NORP', 36: 'S-GPE', 40: 'S-DATE', 44: 'S-FAC', 48: 'S-QUANTITY', 52: 'S-ORDINAL', 56: 'S-TIME', 60: 'S-PRODUCT', 64: 'S-PERCENT', 68: 'S-MONEY', 72: 'S-LAW', 76: 'S-LANGUAGE', 80: 'S-group', 84: 'S-corporation', 88: 'S-creative-work', 92: 'S-product', 96: 'S-Gene_or_gene_product', 100: 'S-Complex', 104: 'S-Cellular_component', 108: 'S-Simple_chemical', 112: 'S-Temporal', 116: 'S-DocumentReference', 120: 'S-Quantity', 124: 'S-Nationality', 128: 'S-MilitaryPlatform', 132: 'S-Weapon', 136: 'S-Money', 140: 'S-CHARACTER', 144: 'S-YEAR', 148: 'S-TITLE', 152: 'S-SONG', 156: 'S-ACTOR', 160: 'S-PLOT', 164: 'S-GENRE', 168: 'S-RATING', 172: 'S-DIRECTOR', 176: 'S-RATINGS_AVERAGE', 180: 'S-TRAILER', 184: 'S-REVIEW', 188: 'S-Restaurant_Name', 192: 'S-Rating', 196: 'S-Hours', 200: 'S-Dish', 204: 'S-Cuisine', 208: 'S-Amenity', 212: 'S-Price'}
        RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20|21[,22]*,23|24|25[,26]*,27|28|29[,30]*,31|32|33[,34]*,35|36|37[,38]*,39|40|41[,42]*,43|44|45[,46]*,47|48|49[,50]*,51|52|53[,54]*,55|56|57[,58]*,59|60|61[,62]*,63|64|65[,66]*,67|68|69[,70]*,71|72|73[,74]*,75|76|77[,78]*,79|80|81[,82]*,83|84|85[,86]*,87|88|89[,90]*,91|92|93[,94]*,95|96|97[,98]*,99|100|101[,102]*,103|104|105[,106]*,107|108|109[,110]*,111|112|113[,114]*,115|116|117[,118]*,119|120|121[,122]*,123|124|125[,126]*,127|128|129[,130]*,131|132|133[,134]*,135|136|137[,138]*,139|140|141[,142]*,143|144|145[,146]*,147|148|149[,150]*,151|152|153[,154]*,155|156|157[,158]*,159|160|161[,162]*,163|164|165[,166]*,167|168|169[,170]*,171|172|173[,174]*,175|176|177[,178]*,179|180|181[,182]*,183|184|185[,186]*,187|188|189[,190]*,191|192|193[,194]*,195|196|197[,198]*,199|200|201[,202]*,203|204|205[,206]*,207|208|209[,210]*,211|212'


    elif model == 'single_conll2003' or model == 'single_conll2003_84b':
        TAG_TO_NUNBER = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'E-ORG': 3, 'S-ORG': 4, 'B-MISC': 5, 'I-MISC': 6, 'E-MISC': 7, 'S-MISC': 8, 'B-PER': 9, 'I-PER': 10, 'E-PER': 11, 'S-PER': 12, 'B-LOC': 13, 'I-LOC': 14, 'E-LOC': 15, 'S-LOC': 16}
        RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16'

    elif model =='single_ontonotes5':
        TAG_TO_NUNBER = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'E-ORG': 3, 'S-ORG': 4, 'B-WORK_OF_ART': 5, 'I-WORK_OF_ART': 6, 'E-WORK_OF_ART': 7, 'S-WORK_OF_ART': 8, 'B-LOC': 9, 'I-LOC': 10, 'E-LOC': 11, 'S-LOC': 12, 'B-CARDINAL': 13, 'I-CARDINAL': 14, 'E-CARDINAL': 15, 'S-CARDINAL': 16, 'B-EVENT': 17, 'I-EVENT': 18, 'E-EVENT': 19, 'S-EVENT': 20, 'B-NORP': 21, 'I-NORP': 22, 'E-NORP': 23, 'S-NORP': 24, 'B-GPE': 25, 'I-GPE': 26, 'E-GPE': 27, 'S-GPE': 28, 'B-DATE': 29, 'I-DATE': 30, 'E-DATE': 31, 'S-DATE': 32, 'B-PERSON': 33, 'I-PERSON': 34, 'E-PERSON': 35, 'S-PERSON': 36, 'B-FAC': 37, 'I-FAC': 38, 'E-FAC': 39, 'S-FAC': 40, 'B-QUANTITY': 41, 'I-QUANTITY': 42, 'E-QUANTITY': 43, 'S-QUANTITY': 44, 'B-ORDINAL': 45, 'I-ORDINAL': 46, 'E-ORDINAL': 47, 'S-ORDINAL': 48, 'B-TIME': 49, 'I-TIME': 50, 'E-TIME': 51, 'S-TIME': 52, 'B-PRODUCT': 53, 'I-PRODUCT': 54, 'E-PRODUCT': 55, 'S-PRODUCT': 56, 'B-PERCENT': 57, 'I-PERCENT': 58, 'E-PERCENT': 59, 'S-PERCENT': 60, 'B-MONEY': 61, 'I-MONEY': 62, 'E-MONEY': 63, 'S-MONEY': 64, 'B-LAW': 65, 'I-LAW': 66, 'E-LAW': 67, 'S-LAW': 68, 'B-LANGUAGE': 69, 'I-LANGUAGE': 70, 'E-LANGUAGE': 71, 'S-LANGUAGE': 72}
        RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20|21[,22]*,23|24|25[,26]*,27|28|29[,30]*,31|32|33[,34]*,35|36|37[,38]*,39|40|41[,42]*,43|44|45[,46]*,47|48|49[,50]*,51|52|53[,54]*,55|56|57[,58]*,59|60|61[,62]*,63|64|65[,66]*,67|68|69[,70]*,71|72'
    elif model =='single_wikigold':

        TAG_TO_NUNBER = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'E-ORG': 3, 'S-ORG': 4, 'B-MISC': 5, 'I-MISC': 6, 'E-MISC': 7,
                         'S-MISC': 8, 'B-PER': 9, 'I-PER': 10, 'E-PER': 11, 'S-PER': 12, 'B-LOC': 13, 'I-LOC': 14,
                         'E-LOC': 15, 'S-LOC': 16}
        RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16'
    elif model =='single_wnut17':
        TAG_TO_NUNBER = {'O': 0, 'B-location': 1, 'I-location': 2, 'E-location': 3, 'S-location': 4, 'B-group': 5, 'I-group': 6, 'E-group': 7, 'S-group': 8, 'B-corporation': 9, 'I-corporation': 10, 'E-corporation': 11, 'S-corporation': 12, 'B-person': 13, 'I-person': 14, 'E-person': 15, 'S-person': 16, 'B-creative-work': 17, 'I-creative-work': 18, 'E-creative-work': 19, 'S-creative-work': 20, 'B-product': 21, 'I-product': 22, 'E-product': 23, 'S-product': 24}

        RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20|21[,22]*,23|24'

    elif model =='single_mitmovieeng':

        TAG_TO_NUNBER= {'O': 0, 'B-CHARACTER': 1, 'I-CHARACTER': 2, 'E-CHARACTER': 3, 'S-CHARACTER': 4, 'B-YEAR': 5, 'I-YEAR': 6, 'E-YEAR': 7, 'S-YEAR': 8, 'B-TITLE': 9, 'I-TITLE': 10, 'E-TITLE': 11, 'S-TITLE': 12, 'B-SONG': 13, 'I-SONG': 14, 'E-SONG': 15, 'S-SONG': 16, 'B-ACTOR': 17, 'I-ACTOR': 18, 'E-ACTOR': 19, 'S-ACTOR': 20, 'B-PLOT': 21, 'I-PLOT': 22, 'E-PLOT': 23, 'S-PLOT': 24, 'B-GENRE': 25, 'I-GENRE': 26, 'E-GENRE': 27, 'S-GENRE': 28, 'B-RATING': 29, 'I-RATING': 30, 'E-RATING': 31, 'S-RATING': 32, 'B-DIRECTOR': 33, 'I-DIRECTOR': 34, 'E-DIRECTOR': 35, 'S-DIRECTOR': 36, 'B-RATINGS_AVERAGE': 37, 'I-RATINGS_AVERAGE': 38, 'E-RATINGS_AVERAGE': 39, 'S-RATINGS_AVERAGE': 40, 'B-TRAILER': 41, 'I-TRAILER': 42, 'E-TRAILER': 43, 'S-TRAILER': 44, 'B-REVIEW': 45, 'I-REVIEW': 46, 'E-REVIEW': 47, 'S-REVIEW': 48}
        RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20|21[,22]*,23|24|25[,26]*,27|28|29[,30]*,31|32|33[,34]*,35|36|37[,38]*,39|40|41[,42]*,43|44|45[,46]*,47|48'

    elif model =='single_bionlp13pc':
        TAG_TO_NUNBER ={'O': 0, 'B-Gene_or_gene_product': 1, 'I-Gene_or_gene_product': 2, 'E-Gene_or_gene_product': 3, 'S-Gene_or_gene_product': 4, 'B-Complex': 5, 'I-Complex': 6, 'E-Complex': 7, 'S-Complex': 8, 'B-Cellular_component': 9, 'I-Cellular_component': 10, 'E-Cellular_component': 11, 'S-Cellular_component': 12, 'B-Simple_chemical': 13, 'I-Simple_chemical': 14, 'E-Simple_chemical': 15, 'S-Simple_chemical': 16}

        RE_PATTENS= r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16'

    elif model == 'single_re3d':
        TAG_TO_NUNBER = {'O': 0, 'B-Organisation': 1, 'I-Organisation': 2, 'E-Organisation': 3, 'S-Organisation': 4, 'B-Location': 5, 'I-Location': 6, 'E-Location': 7, 'S-Location': 8, 'B-Temporal': 9, 'I-Temporal': 10, 'E-Temporal': 11, 'S-Temporal': 12, 'B-Person': 13, 'I-Person': 14, 'E-Person': 15, 'S-Person': 16, 'B-DocumentReference': 17, 'I-DocumentReference': 18, 'E-DocumentReference': 19, 'S-DocumentReference': 20, 'B-Quantity': 21, 'I-Quantity': 22, 'E-Quantity': 23, 'S-Quantity': 24, 'B-Nationality': 25, 'I-Nationality': 26, 'E-Nationality': 27, 'S-Nationality': 28, 'B-MilitaryPlatform': 29, 'I-MilitaryPlatform': 30, 'E-MilitaryPlatform': 31, 'S-MilitaryPlatform': 32, 'B-Weapon': 33, 'I-Weapon': 34, 'E-Weapon': 35, 'S-Weapon': 36, 'B-Money': 37, 'I-Money': 38, 'E-Money': 39, 'S-Money': 40}

        RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20|21[,22]*,23|24|25[,26]*,27|28|29[,30]*,31|32|33[,34]*,35|36|37[,38]*,39|40'


    elif model =='single_mitrestaurant':
        TAG_TO_NUNBER = {'O': 0, 'B-Restaurant_Name': 1, 'I-Restaurant_Name': 2, 'E-Restaurant_Name': 3, 'S-Restaurant_Name': 4, 'B-Rating': 5, 'I-Rating': 6, 'E-Rating': 7, 'S-Rating': 8, 'B-Location': 9, 'I-Location': 10, 'E-Location': 11, 'S-Location': 12, 'B-Hours': 13, 'I-Hours': 14, 'E-Hours': 15, 'S-Hours': 16, 'B-Dish': 17, 'I-Dish': 18, 'E-Dish': 19, 'S-Dish': 20, 'B-Cuisine': 21, 'I-Cuisine': 22, 'E-Cuisine': 23, 'S-Cuisine': 24, 'B-Amenity': 25, 'I-Amenity': 26, 'E-Amenity': 27, 'S-Amenity': 28, 'B-Price': 29, 'I-Price': 30, 'E-Price': 31, 'S-Price': 32}
        RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20|21[,22]*,23|24|25[,26]*,27|28|29[,30]*,31|32'

    elif model =='single_sec':
        TAG_TO_NUNBER = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'E-ORG': 3, 'S-ORG': 4, 'B-MISC': 5, 'I-MISC': 6, 'E-MISC': 7,
                         'S-MISC': 8, 'B-PER': 9, 'I-PER': 10, 'E-PER': 11, 'S-PER': 12, 'B-LOC': 13, 'I-LOC': 14,
                         'E-LOC': 15, 'S-LOC': 16}
        RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16'


    elif model.startswith('ace2005'):
        TAG_TO_NUNBER = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'E-PER': 3, 'S-PER': 4, 'B-GPE': 5, 'I-GPE': 6, 'E-GPE': 7, 'S-GPE': 8, 'B-ORG': 9, 'I-ORG': 10, 'E-ORG': 11, 'S-ORG': 12, 'B-LOC': 13, 'I-LOC': 14, 'E-LOC': 15, 'S-LOC': 16, 'B-WEA': 17, 'I-WEA': 18, 'E-WEA': 19, 'S-WEA': 20, 'B-FAC': 21, 'I-FAC': 22, 'E-FAC': 23, 'S-FAC': 24, 'B-VEH': 25, 'I-VEH': 26, 'E-VEH': 27, 'S-VEH': 28}

        RE_PATTENS = '1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20|21[,22]*,23|24|25[,26]*,27|28'


    else:
        pass

    return TAG_TO_NUNBER, RE_PATTENS



import re
def Print_BOIES_encoding():
    ALL_TAGS = ['ORG', 'MISC', 'PER', 'LOC', 'WORK_OF_ART', 'CARDINAL', 'EVENT', 'NORP', 'GPE', 'DATE', 'FAC', 'QUANTITY', 'ORDINAL', 'TIME', 'PRODUCT', 'PERCENT', 'MONEY', 'LAW', 'LANGUAGE', 'group', 'corporation', 'creative-work', 'product', 'Gene_or_gene_product', 'Complex', 'Cellular_component', 'Simple_chemical', 'Temporal', 'DocumentReference', 'Quantity', 'Nationality', 'MilitaryPlatform', 'Weapon', 'Money', 'CHARACTER', 'YEAR', 'TITLE', 'SONG', 'ACTOR', 'PLOT', 'GENRE', 'RATING', 'DIRECTOR', 'RATINGS_AVERAGE', 'TRAILER', 'REVIEW', 'Restaurant_Name', 'Rating', 'Hours', 'Dish', 'Cuisine', 'Amenity', 'Price']
    ALL_TAGS = ['ORG','MISC','PER','LOC']
    ALL_TAGS = ['ORG', 'WORK_OF_ART', 'LOC', 'CARDINAL', 'EVENT', 'NORP', 'GPE', 'DATE', 'PERSON', 'FAC', 'QUANTITY', 'ORDINAL', 'TIME', 'PRODUCT', 'PERCENT', 'MONEY', 'LAW', 'LANGUAGE']
    ALL_TAGS = ['MISC', 'PER', 'ORG', 'LOC']

    ALL_TAGS= ['location', 'group', 'corporation', 'person', 'creative-work', 'product']

    ALL_TAGS = ['CHARACTER', 'YEAR', 'TITLE', 'SONG', 'ACTOR', 'PLOT', 'GENRE', 'RATING', 'DIRECTOR', 'RATINGS_AVERAGE', 'TRAILER', 'REVIEW']
    ALL_TAGS = ['Gene_or_gene_product', 'Complex', 'Cellular_component', 'Simple_chemical']

    ALL_TAGS = ['Organisation', 'Location', 'Temporal', 'Person', 'DocumentReference', 'Quantity', 'Nationality', 'MilitaryPlatform', 'Weapon', 'Money']

    ALL_TAGS =['Restaurant_Name', 'Rating', 'Location', 'Hours', 'Dish', 'Cuisine', 'Amenity', 'Price']
    ALL_TAGS = ['PER', 'GPE', 'ORG', 'LOC', 'WEA', 'FAC', 'VEH']


    all_mapping_dic = {}

    all_mapping_dic['O']=0

    B_mapping_dic = {}
    E_mapping_dic = {}
    S_mapping_dic ={}

    i = 1

    type = 0
    index_to_type = {}
    patten_list = []
    for t in ALL_TAGS:
        all_mapping_dic['B-'+t] = i
        index_to_type[i]=type

        B_mapping_dic[i] = 'B-'+t

        all_mapping_dic['I-' + t] = i+1
        all_mapping_dic['E-' + t] = i+2
        E_mapping_dic[i+2] = 'E-' + t

        all_mapping_dic['S-' + t] = i+3
        S_mapping_dic[i+3] = 'S-' + t


        index_to_type[i+3] = type

        #r'9[,10]*,11|1[,2]*,3'

        patten_list.append(str(i)+'[,'+str(i+1)+']*,'+str(i+2))
        patten_list.append(str(i+3))

        i  = i +4
        type = type + 1

    RE_PATTENS = '|'.join(patten_list)




    print(all_mapping_dic)
    print(len(all_mapping_dic))
    #print(index_to_type)

    #print(B_mapping_dic)
    #print(E_mapping_dic)
    #print(S_mapping_dic)

    print()
    print(RE_PATTENS)






if __name__ == '__main__':


    Print_BOIES_encoding()