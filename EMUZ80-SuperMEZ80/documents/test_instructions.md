# Test instructions

## Build test

```
% make realclean
% make test_build
```

## IDE test

* Launch MPLAB X IDE 6.15
* File -> Open Project, choose and open mplab.X/
* Production -> Batch Build Project..., check all configurations and push Clean and Build
* Use PPICKit to download the firmware to SuperMEZ80-CPM
* Select and boot CP/M 3.0

## EMUZ80 + SuperMEZ80-CPM

```
% make BOARD=SUPERMEZ80_CPM PIC=18F47Q43 upload
% make BOARD=SUPERMEZ80_CPM PIC=18F47Q43 test test_monitor test_time
=========================
     BOOT TIME:    3.989
  COMPILE TIME:   46.318
 ASCIIART TIME:  181.804
=========================
```

## EMUZ80 + SuperMEZ80-SPI

```
% make BOARD=SUPERMEZ80_SPI PIC=18F47Q43 upload
% make BOARD=SUPERMEZ80_SPI PIC=18F47Q43 test test_monitor test_time
=========================
     BOOT TIME:    4.331
  COMPILE TIME:   51.836
 ASCIIART TIME:  181.285
=========================
```

## EMUZ80-57Q

```
% make BOARD=EMUZ80_57Q PIC=18F57Q43 upload
% make BOARD=EMUZ80_57Q PIC=18F57Q43 test test_monitor test_repeat

OK
```

## Z8S180-57Q

```
% make BOARD=Z8S180_57Q PIC=18F57Q43 upload
% make BOARD=Z8S180_57Q PIC=18F57Q43 test_repeat

OK
```

## EMUZ80 + MEZ80SD

```
% make BOARD=SUPERMEZ80_SPI PIC=18F47Q43 upload
% make BOARD=SUPERMEZ80_SPI PIC=18F47Q43 test_repeat

OK
```
