const DEFAULT_LATITUDE = 40.7128; // Default latitude for example: New York
const DEFAULT_LONGITUDE = -74.0060; // Default longitude for example: New York

// On page load, try to get user's location
getLocation();

function getLocation() 
{
    if (navigator.geolocation) 
    {
        navigator.geolocation.getCurrentPosition(showPosition, showError);
    } else 
    {
        document.getElementById("location").innerHTML = "Geolocation is not supported by this browser.";
    }
}

function showPosition(position) 
{
    var latitude = position.coords.latitude;
    var longitude = position.coords.longitude;
    initMap(latitude, longitude);
}

function showError(error) 
{
    switch(error.code) 
    {
        case error.PERMISSION_DENIED:
            document.getElementById("location").innerHTML = "User denied the request for Geolocation.";
            break;
        case error.POSITION_UNAVAILABLE:
            document.getElementById("location").innerHTML = "Location information is unavailable.";
            break;
        case error.TIMEOUT:
            document.getElementById("location").innerHTML = "The request to get user location timed out.";
            break;
        case error.UNKNOWN_ERROR:
            document.getElementById("location").innerHTML = "An unknown error occurred.";
            break;
    }
    // Default location
    initMap(DEFAULT_LATITUDE, DEFAULT_LONGITUDE);
}

function initMap(lats,longs) 
{
    // Initialize map centered at location
    const map = new google.maps.Map(document.getElementById("map"), {
    center: { lat: lats, lng: longs },
    zoom: 15,
    });

    // Create a PlacesService object to interact with Places API
    const service = new google.maps.places.PlacesService(map);

    // Define the request to search for skin specialists within a radius
    const request = {
        location: { lat: lats, lng: longs }, //location
        radius: 100000, // 100km radius
        keyword: "skin specialist",
    };

    // Perform nearby search to find skin specialists
    service.nearbySearch(request, (results, status) => {
        if (status === google.maps.places.PlacesServiceStatus.OK) 
        {
            // Loop through the results and create markers
            for (let i = 0; i < results.length; i++) 
            {
                createMarker(results[i]);
            }
        }
    });

    // Function to create markers for search results
    function createMarker(place) 
    {
        const marker = new google.maps.Marker({
            map: map,
            position: place.geometry.location,
        });

        google.maps.event.addListener(marker, "click", () => {
            const infowindow = new google.maps.InfoWindow();
            infowindow.setContent(place.name);
            infowindow.open(map, marker);
        });
    }
}

function toggleDropdown() 
{
    var dropdownList = document.getElementById("dropdown-list");
    if (dropdownList.style.display === "none" || dropdownList.style.display === "") 
    {
        dropdownList.style.display = "block";
    } 
    else 
    {
        dropdownList.style.display = "none";
    }
}

document.addEventListener("DOMContentLoaded", function() {
    var dropdownList = document.getElementById("dropdown-list");
    dropdownList.style.display = "none";
});