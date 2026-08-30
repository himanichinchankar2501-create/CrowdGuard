def calculate_density(people_count, width, height):
    area = width * height

    if area == 0:
        return 0.0

    density = (people_count / area) * 100000

    return round(density, 3)


def calculate_risk(density):
    if density < 1.0:
        return "LOW"

    elif density < 2.5:
        return "MODERATE"

    else:
        return "HIGH"