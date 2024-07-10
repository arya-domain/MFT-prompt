def promts(data1Name):
    if data1Name == "Trento":
        classes = {
            0: "annotated apple trees and fruit in hyperspectral images",  # Apples
            1: "identified various manmade structures such as houses commercial buildings and industrial facilities in hyperspectral imagery",  # Buildings
            2: "labeled the earth surface which can include materials like soil grass rocks and natural terrain elements in hyperspectral data",  # Ground
            3: "recognized dense forested areas and enabling analysis of the spectral properties of different tree species and woodland ecosystems in hyperspectral images",  # Woods
            4: "categorized grapevines for monitoring and managing their health and grape quality in hyperspectral images used for wine production",  # Vineyard
            5: "detected paved and unpaved roads for applications such as transportation analysis and infrastructure assessment in hyperspectral images",  # Roads
        }
    if data1Name == "Houston":
        classes = {
            0: "lush green grass thriving and vibrant symbolizing the epitome of natural health in hyperspectral image",  # Healthy grass
            1: "grass exhibiting signs of stress perhaps due to environmental factors or insufficient care in hyperspectral image",  # Stressed grass
            2: "artificial turf resembling natural grass commonly used in various settings for its low maintenance in hyperspectral image",  # Synthetic grass
            3: "tall and majestic trees providing shade and contributing to the ecosystem with their green foliage in hyperspectral image",  # Trees
            4: "the earthy ground a foundation for plant life featuring a mix of minerals and organic matter in hyperspectral image",  # Soil
            5: "a source of life whether it is a serene pond flowing river or any form of liquid sustenance for nature in hyperspectral image",  # Water
            6: "areas designated for housing where communities and families make their homes in hyperspectral image",  # Residential
            7: "spaces designed for business activities from offices to shops fostering economic endeavors in hyperspectral image",  # Commercial
            8: "paved pathways for vehicular transportation connecting destinations and facilitating travel in hyperspectral image",  # Road
            9: "wide and well-traveled roads designed for high-speed long-distance travel between cities and regions in hyperspectral image",  # Highway
            10: "tracks and infrastructure for trains enabling efficient and reliable transportation of goods and passengers in hyperspectral image",  # Railway
            11: "designated areas for parking vehicles providing convenience and organization in hyperspectral image",  # Parking Lot 1
            12: "additional parking space catering to the need for accommodating a larger number of vehicles in hyperspectral image",  # Parking Lot 2
            13: "a sports surface dedicated to tennis featuring marked boundaries and a net for the game in hyperspectral image",  # Tennis Court
            14: "a specially designed track for running often found in sports facilities and schools encouraging physical fitness in hyperspectral image",  # Running Track
        }
    if data1Name == "MUUFL":
        classes = {
            0: "tall and majestic trees providing shade and contributing to the ecosystem with their green foliage in hyperspectral image",  # Trees
            1: "pure grass representing a natural and untarnished ground cover in hyperspectral image in hyperspectral image",  # Grass Pure
            2: "grass on the ground surface depicting the varied textures and patterns of natural landscapes in hyperspectral image",  # Grass Groundsurface
            3: "dirt and sand the granular components of the ground with diverse natural compositions in hyperspectral image",  # Dirt And Sand
            4: "materials used in road construction forming the paved pathways for transportation in hyperspectral image",  # Road Materials
            5: "water a fundamental element in nature whether in the form of lakes rivers or other bodies in hyperspectral image",  # Water
            6: "shadows cast by buildings adding depth and dimension to urban landscapes in hyperspectral image",  # Buildings Shadow
            7: "buildings structures designed for various purposes from residential to commercial in hyperspectral image",  # Buildings
            8: "sidewalks pedestrian pathways along roads and streets facilitating safe walking in hyperspectral image",  # Sidewalk
            9: "yellow curb markings indicating restrictions or specific rules for parking and stopping in hyperspectral image",  # Yellow Curb
            10: "cloth panels possibly used for decorative or functional purposes in outdoor settings in hyperspectral image",  # ClothPanels
        }
    return classes


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(xtest, xtest2, ytest, name, model, tokens):
    pred_y = np.empty((len(ytest)), dtype=np.float32)
    number = len(ytest) // testSizeNumber
    for i in range(number):
        temp = xtest[i * testSizeNumber : (i + 1) * testSizeNumber, :, :]
        temp = temp.cuda()
        temp1 = xtest2[i * testSizeNumber : (i + 1) * testSizeNumber, :, :]
        temp1 = temp1.cuda()
        token_tokens = tokens[i * testSizeNumber : (i + 1) * testSizeNumber,]
        token_tokens = token_tokens.cuda()
        temp2 = model(temp, temp1, token_tokens)

        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i * testSizeNumber : (i + 1) * testSizeNumber] = temp3.cpu()
        del temp, temp2, temp3, temp1

    if (i + 1) * testSizeNumber < len(ytest):
        temp = xtest[(i + 1) * testSizeNumber : len(ytest), :, :]
        temp = temp.cuda()
        temp1 = xtest2[(i + 1) * testSizeNumber : len(ytest), :, :]
        temp1 = temp1.cuda()
        token_tokens = tokens[(i + 1) * testSizeNumber : len(ytest),]
        token_tokens = token_tokens.cuda()
        temp2 = model(temp, temp1, token_tokens)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * testSizeNumber : len(ytest)] = temp3.cpu()
        del temp, temp2, temp3, temp1

    pred_y = torch.from_numpy(pred_y).long()

    if name == "Houston":
        target_names = [
            "Healthy grass",
            "Stressed grass",
            "Synthetic grass",
            "Trees",
            "Soil",
            "Water",
            "Residential",
            "Commercial",
            "Road",
            "Highway",
            "Railway",
            "Parking Lot 1",
            "Parking Lot 2",
            "Tennis Court",
            "Running Track",
        ]

        #

    elif name == "Trento":
        target_names = ["Apples", "Buildings", "Ground", "Woods", "Vineyard", "Roads"]
    elif name == "MUUFL" or name == "MUUFLS" or name == "MUUFLSR":
        target_names = [
            "Trees",
            "Grass_Pure",
            "Grass_Groundsurface",
            "Dirt_And_Sand",
            "Road_Materials",
            "Water",
            "Buildings'_Shadow",
            "Buildings",
            "Sidewalk",
            "Yellow_Curb",
            "ClothPanels",
        ]


    #     classification = classification_report(ytest, pred_y, target_names=target_names)
    oa = accuracy_score(ytest, pred_y)
    confusion = confusion_matrix(ytest, pred_y)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(ytest, pred_y)

    return confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def seed_val(set):
    if set == "Trento":
        return 15 if HSIOnly else 50
    if set == "Houston":
        return 155 if HSIOnly else 119
    if set == "MUUFL":
        return 11 if HSIOnly else 9


def normalize_val(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

