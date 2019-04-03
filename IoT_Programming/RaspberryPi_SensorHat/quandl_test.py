#o-n5ZDbb-ewagY5j8VYf
import quandl
import requests

data = quandl.get("XJPX/13050", authtoken="o-n5ZDbb-ewagY5j8VYf")
print(data[:5])

#COLUMN    DESCRIPTION
#Date    
#Open    Stock price at market open
#High    Highest price of the day
#Low    Lowest price of the day
#Close    Stock price at market close
#Volume    Volume of trades for the day
#Adjustment Factor    The factor by which historical share prices/volumes are adjusted
#Adjustment Type    A numeric code (integer) corresponding to the corporate action that precipitated adjustment, such as dividend, consolidation, etc. If more than one corporate action occurs for the day, the individual codes are combined. For more details, see the Adjustment Types section below.
