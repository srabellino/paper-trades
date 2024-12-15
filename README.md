# alpacaOptionsBacktest
 Backtesting and trading options through alpaca

 #--------------- TODO -----------------#
 + look into why some db entries have the timezone and some don't

 #-------------- Updates ---------------#
 9/26
 Used chatgpt to convert to OOP code and added logging and comments
 Modified the find target delta function to fix a bug involving put contracts
 needing to use the absolute value of the delta number
 Added a try statemtnt to handle when the greeks aren't included in the option
 date
 Tested and deployed to Lambda

 9/16
 Changed the options data to come from the snapshot api instead of the 
 alpaca sdk.
 Changed the market order to a limit order for purchasing the contracts, and
 added a function to determine the mid price between the most recent bid
 and ask from the snapshot api
