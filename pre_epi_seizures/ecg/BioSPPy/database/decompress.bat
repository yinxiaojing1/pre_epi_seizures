@echo off

rem *** runPhilipsXMLECGdecomp.bat ***
rem *** version 1.1 10/11/04 EDH ***

rem *** This script runs the Philips ECG Waveform Decompressor on all the Philips
rem        XML ECG files in a folder to create Philips XML ECG files
rem        with uncompressed, plain text, waveform sample values.

rem ***************************************************************************************
rem *** user may edit these:

rem * NOTE: all paths below must NOT end in a backslash ("\") 

rem * Path to input folder

rem use this to run on ECGs in the program install folder, PhilipsXMLECGs sub-folder:
SET INPUT_FOLDER=.\20140520
rem Use this to run on ECGs in a sub-folder named "PhilipsXMLECGs" below the current folder
rem SET INPUT_FOLDER=.\PhilipsXMLECGs

rem *   Use this to run on ECGs in the current folder:
rem SET INPUT_FOLDER=.
rem *   Another example:
rem SET INPUT_FOLDER=C:\PhilipsXMLECGs


rem * Path to output folder
rem * Note: This must be a ABOSLUTE path (e.g., C:\PhilipsDecompressed),
rem *   or a RELATIVE path RELATIVE TO THE INPUT_FOLDER !
rem * This creates an "Decompressed" sub-folder in the input ECG folder:
SET OUTPUT_FOLDER=.\Decompressed

rem * A copy of the input file will be placed in this folder unless the path is "noCopy"
rem * Note: If not "noCopy", this must be a ABSOLUTE path (e.g., C:\Decompressed_originals),
rem *   or a RELATIVE path RELATIVE TO THE INPUT_FOLDER !
rem * (This program feature is useful since the directory scanner in TracemasterVue Rel 1
rem *     deletes the input ECG file after calling the program.)
SET COPY_INPUT_FILE_PATH=noCopy
rem SET COPY_INPUT_FILE_PATH=C:\TracemasterVue\AddOns\Decompressed_originals

rem *This string is placed at the end of the output filename
SET OUTPUT_FILENAME_SUFFIX=.decomp.xml

rem * If "blob", the rhythm waveforms will be in one parsedwaveforms section
rem * if "leadBylead", the rhythm waveforms will be in the leadwaveforms section
rem      with each lead in a separate node
SET OUTPUT_FORMAT=blob
rem SET OUTPUT_FORMAT=leadBylead


rem *** end of user edits
rem ***************************************************************************************

rem PhilipsXMLECGdecompressor usage: inputFileName [inputPath [outputPath [copyInputPath | \"noCopy\" [outputFilenameSuffix [\"blob\" | \"leadBylead\"]]]]]

rem *** this is the name of the program to run
SET PROGRAM=PhilipsXMLECGdecompressor.exe

rem *** if the path to the program does not exist in an environment variable: Error 
if "%PhilipsXMLECGdecomp_PATH%." == "." goto NO_PATH

rem *** if the input folder does not exist, error
if not exist "%INPUT_FOLDER%" goto NO_INPUT_FOLDER

rem *** Everything seems to be OK, lets go ...

rem * change directory to the input folder
echo Changing directory to:
echo "%INPUT_FOLDER%" ...
echo ...
cd "%INPUT_FOLDER%"

rem *** if the output folder does not exist, create it
if not exist "%OUTPUT_FOLDER%" mkdir "%OUTPUT_FOLDER%"


rem *** run program on each xml file in the current folder

FOR %%f IN ( "*.xml" ) DO CALL "%PhilipsXMLECGdecomp_PATH%\%PROGRAM%" "%%f"  "."  "%OUTPUT_FOLDER%"  "%COPY_INPUT_FILE_PATH%"  "%OUTPUT_FILENAME_SUFFIX%"  "%OUTPUT_FORMAT%" 

goto END

:NO_PATH
echo Error: The system "PhilipsXMLECGdecomp_PATH" environment
echo  variable does not point to the PhilipsXMLECGdecompressor installation folder.
echo If you have just installed the program, you need to log off and
echo   log back on, or restart your computer.
echo Otherwise, please re-install the program and/or check that the
echo   environment variable is present.
pause
goto END

:NO_INPUT_FOLDER
echo Error: The Input folder specified, "%INPUT_FOLDER%", does not exist.
echo Please be sure it is specified correctly.
pause
goto END


:END
pause

