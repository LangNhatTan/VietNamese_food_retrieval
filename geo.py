import winrt.windows.devices.geolocation as wdg
from geopy.geocoders import Nominatim
import asyncio

async def getCoords(): 
    locator = wdg.Geolocator()  
    pos = await locator.get_geoposition_async()
    return [pos.coordinate.latitude, pos.coordinate.longitude] 
def getLoc():
    return asyncio.run(getCoords())

def getLocation():
    lat, lng = asyncio.run(getCoords())
    geolocator = Nominatim(user_agent="my_geopy_app")
    location = geolocator.reverse(str(lat) + "," + str(lng))
    return lat, lng, location

def getAddress(lat, logt):
    geolocator = Nominatim(user_agent = "my_app")
    location = geolocator.reverse(str(lat) + "," + str(logt))
    return location

