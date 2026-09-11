from environment.city import City

for i in range(5):
    city = City(seed=i)

    print(f"\nEpisode {i + 1}")
    print("Buildings:", city.buildings)
    print("No-fly:", city.no_fly_zones)
    print("Warehouses:", city.warehouses)
    print("Delivery:", city.delivery_points)
    print("Charging:", city.charging_stations)