#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misumi XY Controller Wrapper
----------------------------
A Python wrapper for the Misumi DS102/DS112 Series Stepping Motor Controller.
This module provides a high-level interface to control the Misumi XY stage
via serial communication (RS232C or USB).

Based on the DS102/DS112 Series Operation Manual Ver 2.00
"""

import serial
import time
import logging
from enum import Enum, auto
from typing import Union, List, Dict, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MisumiXYWrapper")


class AxisName(Enum):
    """Enum for axis names"""
    X = 1
    Y = 2
    Z = 3
    U = 4
    V = 5
    W = 6
    ALL = "ALL"


class Direction(Enum):
    """Enum for direction"""
    CW = 0
    CCW = 1


class DriveMode(Enum):
    """Enum for drive modes"""
    CW = 0
    CCW = 1
    ORIGIN = 2
    HOME = 3
    ABS = 4
    CWJ = 5  # Jog drive to CW
    CCWJ = 6  # Jog drive to CCW


class StopMode(Enum):
    """Enum for stop modes"""
    EMERGENCY = 0
    REDUCTION = 1


class UnitType(Enum):
    """Enum for unit types"""
    PULSE = 0
    UM = 1  # μm
    MM = 2  # mm
    DEG = 3  # deg
    MRAD = 4  # mrad


class OriginReturnType(Enum):
    """Enum for origin return types"""
    TYPE0 = 0  # Origin return is not implemented (default)
    TYPE1 = 1  # Start to detect to the CCW, Detect the CW side edge of NORG signal, then Detect the CCW side edge of ORG signal
    TYPE2 = 2  # Start to detect to the CW, Detect the CCW side edge of NORG signal, then Detect the CW side edge of ORG signal
    TYPE3 = 3  # Start to detect to the CCW, Detect the CCW side edge of ORG signal
    TYPE4 = 4  # Start to detect to the CW, Detect the CW side edge of ORG signal
    TYPE5 = 5  # Start to detect to the CCW, Detect the CW side edge of CCWLS signal
    TYPE6 = 6  # Start to detect to the CW, Detect the CCW side edge of CWLS signal
    TYPE7 = 7  # After operated type1, detect CCW side edge of TIMING signal
    TYPE8 = 8  # After operated type2, detect CW side edge of TIMING signal
    TYPE9 = 9  # After operated type3, detect CCW side edge of TIMING signal
    TYPE10 = 10  # After operated type4, detect CW side edge of TIMING signal
    TYPE11 = 11  # After operated type5, detect CCW side edge of TIMING signal
    TYPE12 = 12  # After operated type6, detect CW side edge of TIMING signal


class SensorLogic(Enum):
    """Enum for sensor logic"""
    B_NORMAL_CLOSE = 0  # B point (Normal Close)
    A_NORMAL_OPEN = 1  # A point (Normal Open)


class MisumiXYWrapper:
    """
    A wrapper class for the Misumi DS102/DS112 Series Stepping Motor Controller.
    
    This class provides methods to control the Misumi XY stage via serial communication.
    It implements the communication protocol described in the DS102/DS112 Series Operation Manual.
    """

    def __init__(self, port: str, baudrate: int = 38400, timeout: float = 1.0):
        """
        Initialize the MisumiXYWrapper.
        
        Args:
            port (str): Serial port name (e.g., 'COM1', '/dev/ttyUSB0')
            baudrate (int, optional): Baud rate. Defaults to 38400.
            timeout (float, optional): Serial timeout in seconds. Defaults to 1.0.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.connected = False
        self.delimiter = '\r'  # CR (Hex 0D)
        self.connect()
        self.initialize()
    
    def initialize(self):
        self.set_memory_switch_0(1,3)
        self.set_memory_switch_0(2,3)
        self.select_speed(1,6)
        self.select_speed(2,6)
        self.drive(1,2)
        self.drive(2,2)

        while(self.is_in_motion(1) or self.is_in_motion(2)):
            pass

        self.set_position(1,0)
        self.set_position(2,0)

        self.set_home_position(1,0)
        self.set_home_position(2,0)

    def connect(self) -> bool:
        """
        Connect to the controller.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            self.connected = True
            logger.info(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the controller.
        
        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        if self.serial and self.serial.is_open:
            try:
                self.serial.close()
                self.connected = False
                logger.info("Disconnected from controller")
                return True
            except Exception as e:
                logger.error(f"Failed to disconnect: {e}")
                return False
        return True
    
    def _send_command(self, command: str) -> str:
        """
        Send a command to the controller and return the response.
        
        Args:
            command (str): Command to send
            
        Returns:
            str: Response from the controller
        
        Raises:
            ConnectionError: If not connected to the controller
            TimeoutError: If no response is received within timeout
            ValueError: If the response indicates an error
        """
        if not self.connected or not self.serial:
            raise ConnectionError("Not connected to controller")
        
        # Add delimiter if not present
        if not command.endswith(self.delimiter):
            command += self.delimiter
        
        try:
            # Clear input buffer
            self.serial.reset_input_buffer()
            
            # Send command
            logger.debug(f"Sending command: {command.strip()}")
            self.serial.write(command.encode())
            
            # Read response
            response = self.serial.read_until(self.delimiter.encode()).decode().strip()
            logger.debug(f"Received response: {response}")
            
            # Check for error responses
            if response.startswith('E'):
                error_code = response
                error_messages = {
                    'E00': "Stage is not connected or sensor logic setting error",
                    'E01': "Axis is in motion",
                    'E02': "Limit detected",
                    'E03': "Emergency detected",
                    'E20': "Command rule error",
                    'E21': "Error of unsent delimiter",
                    'E22': "Setting range error",
                    'E40': "Communication error",
                    'E41': "Error of write in flash memory"
                }
                error_msg = error_messages.get(error_code, f"Unknown error: {error_code}")
                raise ValueError(f"Controller error: {error_msg}")
            
            return response
        
        except serial.SerialTimeoutException:
            raise TimeoutError("Timeout waiting for response from controller")
        except Exception as e:
            logger.error(f"Communication error: {e}")
            raise
    
    def _format_value(self, value: Union[int, float]) -> str:
        """
        Format a value for sending to the controller.
        
        Args:
            value (Union[int, float]): Value to format
            
        Returns:
            str: Formatted value
        """
        if isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            # Remove trailing zeros after decimal point
            return str(value).rstrip('0').rstrip('.') if '.' in str(value) else str(value)
        else:
            return str(value)
    
    # -------------------------------------------------------------------------
    # Axis Selection and Parameter Setting Commands
    # -------------------------------------------------------------------------
    
    def select_axis(self, axis: Union[AxisName, int, str]) -> None:
        """
        Select an axis for subsequent commands.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to select (1-6 or X, Y, Z, U, V, W or ALL)
        """
        if isinstance(axis, AxisName):
            axis_value = axis.value
        elif isinstance(axis, int) and 1 <= axis <= 6:
            axis_value = axis
        elif isinstance(axis, str) and axis.upper() in ['X', 'Y', 'Z', 'U', 'V', 'W', 'ALL']:
            axis_value = axis.upper()
        else:
            raise ValueError("Invalid axis. Must be 1-6 or X, Y, Z, U, V, W or ALL")
        
        self._send_command(f"AXI{axis_value}")
    
    def set_cw_soft_limit(self, axis: Union[AxisName, int, str], enable: bool, position: Optional[float] = None) -> None:
        """
        Set CW soft limit for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            enable (bool): Enable or disable soft limit
            position (Optional[float]): Position of soft limit (if enable is True)
        """
        self.select_axis(axis)
        self._send_command(f":CWSLE {1 if enable else 0}")
        
        if enable and position is not None:
            self._send_command(f":CWSLP {self._format_value(position)}")
    
    def set_ccw_soft_limit(self, axis: Union[AxisName, int, str], enable: bool, position: Optional[float] = None) -> None:
        """
        Set CCW soft limit for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            enable (bool): Enable or disable soft limit
            position (Optional[float]): Position of soft limit (if enable is True)
        """
        self.select_axis(axis)
        self._send_command(f":CCWSLE {1 if enable else 0}")
        
        if enable and position is not None:
            self._send_command(f":CCWSLP {self._format_value(position)}")
    
    def set_driver_division(self, axis: Union[AxisName, int, str], division: int) -> None:
        """
        Set driver division for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            division (int): Division value (0-15)
                0: 1/1 (Full) division
                1: 1/2 (Half) division
                2: 1/2.5 division
                3: 1/4 division
                4: 1/5 division
                5: 1/8 division
                6: 1/10 division
                7: 1/20 division
                8: 1/25 division
                9: 1/40 division
                10: 1/50 division
                11: 1/80 division
                12: 1/100 division
                13: 1/125 division
                14: 1/200 division
                15: 1/250 division
        """
        if not 0 <= division <= 15:
            raise ValueError("Division must be between 0 and 15")
        
        self.select_axis(axis)
        self._send_command(f":DRDIV {division}")
    
    def set_data_selection(self, axis: Union[AxisName, int, str], data_selection: int) -> None:
        """
        Set DATA selection for the specified axis (only for MS driver).
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            data_selection (int): DATA selection (1 or 2)
                1: DATA1 selection
                2: DATA2 selection
        """
        if data_selection not in [1, 2]:
            raise ValueError("Data selection must be 1 or 2")
        
        self.select_axis(axis)
        self._send_command(f":DATA {data_selection}")
    
    def set_home_position(self, axis: Union[AxisName, int, str], position: float) -> None:
        """
        Set home position for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            position (float): Home position
        """
        self.select_axis(axis)
        self._send_command(f":HOMEP {self._format_value(position)}")
    
    def set_position(self, axis: Union[AxisName, int, str], position: float) -> None:
        """
        Set current position for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            position (float): Current position
        """
        self.select_axis(axis)
        self._send_command(f":POS {self._format_value(position)}")
    
    def set_pulse(self, axis: Union[AxisName, int, str], pulse: float) -> None:
        """
        Set constant step pulse distance for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            pulse (float): Pulse distance
        """
        self.select_axis(axis)
        self._send_command(f":PULS {self._format_value(pulse)}")
    
    def set_pulse_absolute(self, axis: Union[AxisName, int, str], position: float) -> None:
        """
        Set absolute position for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            position (float): Absolute position
        """
        self.select_axis(axis)
        self._send_command(f":PULSA {self._format_value(position)}")
    
    def select_speed(self, axis: Union[AxisName, int, str], speed_table: int) -> None:
        """
        Select speed table for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            speed_table (int): Speed table number (0-9)
        """
        if not 0 <= speed_table <= 9:
            raise ValueError("Speed table must be between 0 and 9")
        
        self.select_axis(axis)
        self._send_command(f":SELSP {speed_table}")
    
    def set_standard_resolution(self, axis: Union[AxisName, int, str], resolution: float) -> None:
        """
        Set standard resolution (distance per pulse at full step) for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            resolution (float): Standard resolution
        """
        self.select_axis(axis)
        self._send_command(f":STANDARD {self._format_value(resolution)}")
    
    def set_unit(self, axis: Union[AxisName, int, str], unit: Union[UnitType, int, str]) -> None:
        """
        Set unit for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            unit (Union[UnitType, int, str]): Unit type
                0 or PULSE: pulse
                1 or UM: μm
                2 or MM: mm
                3 or DEG: deg
                4 or MRAD: mrad
        """
        if isinstance(unit, UnitType):
            unit_value = unit.value
        elif isinstance(unit, int) and 0 <= unit <= 4:
            unit_value = unit
        elif isinstance(unit, str) and unit.upper() in ['PULSE', 'PULS', 'UM', 'MM', 'DEG', 'MRAD']:
            unit_map = {'PULSE': 0, 'PULS': 0, 'UM': 1, 'MM': 2, 'DEG': 3, 'MRAD': 4}
            unit_value = unit_map[unit.upper()]
        else:
            raise ValueError("Invalid unit. Must be 0-4 or PULSE, UM, MM, DEG, MRAD")
        
        self.select_axis(axis)
        self._send_command(f":UNIT {unit_value}")
    
    def set_teaching_point(self, point_number: int, positions: Dict[Union[AxisName, int, str], Union[float, str]]) -> None:
        """
        Set teaching point.
        
        Args:
            point_number (int): Teaching point number (0-63)
            positions (Dict[Union[AxisName, int, str], Union[float, str]]): Dictionary of axis positions
                Key: Axis (1-6 or X, Y, Z, U, V, W)
                Value: Position or 'N' for no data or 'S' for current position
        """
        if not 0 <= point_number <= 63:
            raise ValueError("Teaching point number must be between 0 and 63")
        
        # Initialize position values for all axes
        axis_values = ['N', 'N', 'N', 'N', 'N', 'N']
        
        # Update position values based on provided positions
        for axis, position in positions.items():
            if isinstance(axis, AxisName):
                axis_index = axis.value - 1  # Convert to 0-based index
            elif isinstance(axis, int) and 1 <= axis <= 6:
                axis_index = axis - 1
            elif isinstance(axis, str) and axis.upper() in ['X', 'Y', 'Z', 'U', 'V', 'W']:
                axis_index = {'X': 0, 'Y': 1, 'Z': 2, 'U': 3, 'V': 4, 'W': 5}[axis.upper()]
            else:
                raise ValueError("Invalid axis. Must be 1-6 or X, Y, Z, U, V, W")
            
            if position == 'N' or position == 'S':
                axis_values[axis_index] = position
            else:
                axis_values[axis_index] = self._format_value(position)
        
        # Format the command
        positions_str = '/'.join(axis_values)
        self._send_command(f"TCH{point_number:02d} {positions_str}")
    
    # -------------------------------------------------------------------------
    # Memory Switch Setting Commands
    # -------------------------------------------------------------------------
    
    def set_memory_switch_0(self, axis: Union[AxisName, int, str], origin_type: Union[OriginReturnType, int]) -> None:
        """
        Set memory switch 0 (origin return type) for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            origin_type (Union[OriginReturnType, int]): Origin return type (0-12)
        """
        if isinstance(origin_type, OriginReturnType):
            type_value = origin_type.value
        elif isinstance(origin_type, int) and 0 <= origin_type <= 12:
            type_value = origin_type
        else:
            raise ValueError("Origin return type must be between 0 and 12")
        
        self.select_axis(axis)
        self._send_command(f":MEMSW0 {type_value}")
    
    def set_memory_switch_1(self, axis: Union[AxisName, int, str], limit_logic: Union[SensorLogic, int]) -> None:
        """
        Set memory switch 1 (limit sensor input logic) for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            limit_logic (Union[SensorLogic, int]): Limit sensor logic
                0: B point (Normal Close)
                1: A point (Normal Open)
        """
        if isinstance(limit_logic, SensorLogic):
            logic_value = limit_logic.value
        elif isinstance(limit_logic, int) and limit_logic in [0, 1]:
            logic_value = limit_logic
        else:
            raise ValueError("Limit sensor logic must be 0 or 1")
        
        self.select_axis(axis)
        self._send_command(f":MEMSW1 {logic_value}")
    
    def set_memory_switch_2(self, axis: Union[AxisName, int, str], origin_logic: Union[SensorLogic, int]) -> None:
        """
        Set memory switch 2 (origin sensor input logic) for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            origin_logic (Union[SensorLogic, int]): Origin sensor logic
                0: B point (Normal Close)
                1: A point (Normal Open)
        """
        if isinstance(origin_logic, SensorLogic):
            logic_value = origin_logic.value
        elif isinstance(origin_logic, int) and origin_logic in [0, 1]:
            logic_value = origin_logic
        else:
            raise ValueError("Origin sensor logic must be 0 or 1")
        
        self.select_axis(axis)
        self._send_command(f":MEMSW2 {logic_value}")
    
    def set_memory_switch_3(self, axis: Union[AxisName, int, str], near_origin_logic: Union[SensorLogic, int]) -> None:
        """
        Set memory switch 3 (near origin sensor input logic) for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            near_origin_logic (Union[SensorLogic, int]): Near origin sensor logic
                0: B point (Normal Close)
                1: A point (Normal Open)
        """
        if isinstance(near_origin_logic, SensorLogic):
            logic_value = near_origin_logic.value
        elif isinstance(near_origin_logic, int) and near_origin_logic in [0, 1]:
            logic_value = near_origin_logic
        else:
            raise ValueError("Near origin sensor logic must be 0 or 1")
        
        self.select_axis(axis)
        self._send_command(f":MEMSW3 {logic_value}")
    
    def set_memory_switch_4(self, axis: Union[AxisName, int, str], current_down: bool) -> None:
        """
        Set memory switch 4 (current down control) for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            current_down (bool): Current down control
                True: Control current down
                False: No current down control (only for MS type)
        """
        self.select_axis(axis)
        self._send_command(f":MEMSW4 {0 if current_down else 1}")
    
    def set_memory_switch_5(self, axis: Union[AxisName, int, str], direction: bool) -> None:
        """
        Set memory switch 5 (motion direction switching) for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            direction (bool): Direction
                True: Normal direction (POSITIVE)
                False: Reversed direction (NEGATIVE)
        """
        self.select_axis(axis)
        self._send_command(f":MEMSW5 {0 if direction else 1}")
    
    def set_memory_switch_6(self, axis: Union[AxisName, int, str], stop_type: bool) -> None:
        """
        Set memory switch 6 (stop processing) for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            stop_type (bool): Stop type
                True: Fast stop
                False: Slowdown stop
        """
        self.select_axis(axis)
        self._send_command(f":MEMSW6 {0 if stop_type else 1}")
    
    def set_memory_switch_7(self, axis: Union[AxisName, int, str], zero_reset: bool) -> None:
        """
        Set memory switch 7 (reset after origin return) for the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to set
            zero_reset (bool): Zero reset
                True: Reset to 0 after origin return
                False: No reset after origin return
        """
        self.select_axis(axis)
        self._send_command(f":MEMSW7 {0 if zero_reset else 1}")
    
    # -------------------------------------------------------------------------
    # Speed Table Setting Commands
    # -------------------------------------------------------------------------
    
    def set_speed_table(self, table_number: int, start_speed: int, drive_speed: int, 
                        accel_decel_time: int, s_rate: int = 0) -> None:
        """
        Set speed table parameters.
        
        Args:
            table_number (int): Speed table number (0-9)
            start_speed (int): Start-up speed (1-9999 pps)
            drive_speed (int): Drive speed (1-999999 pps)
            accel_decel_time (int): Acceleration and deceleration time (1-9999 msec)
            s_rate (int, optional): S-curve rate (0-100 %). Defaults to 0.
        """
        if not 0 <= table_number <= 9:
            raise ValueError("Speed table number must be between 0 and 9")
        if not 1 <= start_speed <= 9999:
            raise ValueError("Start-up speed must be between 1 and 9999")
        if not 1 <= drive_speed <= 999999:
            raise ValueError("Drive speed must be between 1 and 999999")
        if not 1 <= accel_decel_time <= 9999:
            raise ValueError("Acceleration and deceleration time must be between 1 and 9999")
        if not 0 <= s_rate <= 100:
            raise ValueError("S-curve rate must be between 0 and 100")
        
        self._send_command(f":L{table_number} {start_speed}")
        self._send_command(f":F{table_number} {drive_speed}")
        self._send_command(f":R{table_number} {accel_decel_time}")
        self._send_command(f":S{table_number} {s_rate}")
    
    # -------------------------------------------------------------------------
    # Write and Reset Commands
    # -------------------------------------------------------------------------
    
    def write_to_flash(self) -> None:
        """
        Write all parameters to flash memory.
        
        Note: Do not power off for over 130 msec after sending this command.
        """
        self._send_command("WRITE")
        # Wait for the write operation to complete
        time.sleep(0.2)
    
    def reset_all_parameters(self) -> None:
        """
        Reset all parameters to default values.
        
        Note: Do not power off for over 5 seconds after sending this command.
        """
        self._send_command("*RST")
        # Wait for the reset operation to complete
        time.sleep(5.5)
    
    # -------------------------------------------------------------------------
    # Driving Commands
    # -------------------------------------------------------------------------
    
    def drive(self, axis: Union[AxisName, int, str], mode: Union[DriveMode, int, str]) -> None:
        """
        Drive the specified axis in the specified mode.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to drive
            mode (Union[DriveMode, int, str]): Drive mode
                0 or CW: Drive in CW direction
                1 or CCW: Drive in CCW direction
                2 or ORIGIN or ORG: Origin return
                3 or HOME: Move to home position
                4 or ABS: Move to absolute position
                5 or CWJ: Jog drive in CW direction
                6 or CCWJ: Jog drive in CCW direction
        """
        if isinstance(mode, DriveMode):
            mode_value = mode.value
        elif isinstance(mode, int) and 0 <= mode <= 6:
            mode_value = mode
        elif isinstance(mode, str):
            mode_map = {
                'CW': 0, 'CCW': 1, 'ORIGIN': 2, 'ORG': 2, 'HOME': 3, 
                'ABS': 4, 'CWJ': 5, 'CCWJ': 6
            }
            if mode.upper() in mode_map:
                mode_value = mode_map[mode.upper()]
            else:
                raise ValueError("Invalid drive mode")
        else:
            raise ValueError("Invalid drive mode")
        
        self.select_axis(axis)
        self._send_command(f":GO {mode_value}")
    
    def drive_absolute(self, axis: Union[AxisName, int, str], position: float) -> None:
        """
        Drive the specified axis to the absolute position.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to drive
            position (float): Absolute position
        """
        self.select_axis(axis)
        self._send_command(f":GOABS {self._format_value(position)}")
    
    def drive_to_teaching_point(self, point_number: int) -> None:
        """
        Drive to the specified teaching point.
        
        Args:
            point_number (int): Teaching point number (0-63)
        """
        if not 0 <= point_number <= 63:
            raise ValueError("Teaching point number must be between 0 and 63")
        
        self._send_command(f"GOTCH {point_number}")
    
    def drive_linear_incremental(self, axis_directions: Dict[Union[AxisName, int, str], bool]) -> None:
        """
        Drive in linear interpolation mode (incremental).
        
        Args:
            axis_directions (Dict[Union[AxisName, int, str], bool]): Dictionary of axis directions
                Key: Axis (X, Y, Z, U, V, W)
                Value: Direction (True for CW/+, False for CCW/-)
        """
        command = "GOLI "
        
        for axis, direction in axis_directions.items():
            if isinstance(axis, AxisName):
                axis_name = axis.name
            elif isinstance(axis, int) and 1 <= axis <= 6:
                axis_name = ['X', 'Y', 'Z', 'U', 'V', 'W'][axis - 1]
            elif isinstance(axis, str) and axis.upper() in ['X', 'Y', 'Z', 'U', 'V', 'W']:
                axis_name = axis.upper()
            else:
                raise ValueError("Invalid axis. Must be 1-6 or X, Y, Z, U, V, W")
            
            command += f"{axis_name}{'+' if direction else '-'}"
        
        self._send_command(command)
    
    def drive_linear_absolute(self, axis_positions: Dict[Union[AxisName, int, str], float]) -> None:
        """
        Drive in linear interpolation mode (absolute).
        
        Args:
            axis_positions (Dict[Union[AxisName, int, str], float]): Dictionary of axis positions
                Key: Axis (X, Y, Z, U, V, W)
                Value: Absolute position
        """
        command = "GOLA "
        
        for axis, position in axis_positions.items():
            if isinstance(axis, AxisName):
                axis_name = axis.name
            elif isinstance(axis, int) and 1 <= axis <= 6:
                axis_name = ['X', 'Y', 'Z', 'U', 'V', 'W'][axis - 1]
            elif isinstance(axis, str) and axis.upper() in ['X', 'Y', 'Z', 'U', 'V', 'W']:
                axis_name = axis.upper()
            else:
                raise ValueError("Invalid axis. Must be 1-6 or X, Y, Z, U, V, W")
            
            command += f"{axis_name}{self._format_value(position)}_"
        
        # Remove trailing underscore
        if command.endswith('_'):
            command = command[:-1]
        
        self._send_command(command)
    
    def stop(self, axis: Optional[Union[AxisName, int, str]] = None, mode: Union[StopMode, int, str] = StopMode.EMERGENCY) -> None:
        """
        Stop the specified axis or all axes.
        
        Args:
            axis (Optional[Union[AxisName, int, str]], optional): Axis to stop. If None, stop all axes. Defaults to None.
            mode (Union[StopMode, int, str], optional): Stop mode. Defaults to StopMode.EMERGENCY.
                0 or EMERGENCY: Emergency stop
                1 or REDUCTION: Slowdown stop
        """
        if isinstance(mode, StopMode):
            mode_value = mode.value
        elif isinstance(mode, int) and mode in [0, 1]:
            mode_value = mode
        elif isinstance(mode, str):
            mode_map = {'EMERGENCY': 0, 'E': 0, 'REDUCTION': 1, 'R': 1}
            if mode.upper() in mode_map:
                mode_value = mode_map[mode.upper()]
            else:
                raise ValueError("Invalid stop mode")
        else:
            raise ValueError("Invalid stop mode")
        
        if axis is None:
            # Stop all axes
            self._send_command(f"STOP_{mode_value}")
        else:
            # Stop specific axis
            self.select_axis(axis)
            self._send_command(f":STOP_{mode_value}")
    
    # -------------------------------------------------------------------------
    # Status Request Commands
    # -------------------------------------------------------------------------
    
    def get_position(self, axis: Union[AxisName, int, str]) -> float:
        """
        Get current position of the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to get position
            
        Returns:
            float: Current position
        """
        self.select_axis(axis)
        response = self._send_command(":POS?")
        return float(response)
    
    def get_status(self, axis: Union[AxisName, int, str]) -> Dict[str, bool]:
        """
        Get status of the specified axis.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to get status
            
        Returns:
            Dict[str, bool]: Status dictionary
        """
        self.select_axis(axis)
        
        # Get status binary 1
        sb1 = int(self._send_command(":SB1?"))
        # Get status binary 2
        sb2 = int(self._send_command(":SB2?"))
        # Get status binary 3
        sb3 = int(self._send_command(":SB3?"))
        
        status = {
            "program_driving": bool(sb1 & 0b10000000),
            "in_motion": bool(sb1 & 0b01000000),
            "home_position_detected": bool(sb1 & 0b00100000),
            "origin_detected": bool(sb1 & 0b00010000),
            "discontinued": bool(sb1 & 0b00001000),
            "soft_limit_detected": bool(sb1 & 0b00000100),
            "mechanical_limit_detected": bool(sb1 & 0b00000010),
            "direction_cw": bool(sb1 & 0b00000001),
            
            "cw_mechanical_limit_detected": bool(sb2 & 0b00000001),
            "ccw_mechanical_limit_detected": bool(sb2 & 0b00000010),
            "cw_soft_limit_detected": bool(sb2 & 0b00000100),
            "ccw_soft_limit_detected": bool(sb2 & 0b00001000),
            "cw_soft_limit_enabled": bool(sb2 & 0b00010000),
            "ccw_soft_limit_enabled": bool(sb2 & 0b00100000),
            
            "axis_selection_available": bool(sb3 & 0b00000001),
            "micro_step_driver": bool(sb3 & 0b00000010) or bool(sb3 & 0b00010000),
        }
        
        return status
    
    def is_in_motion(self, axis: Union[AxisName, int, str]) -> bool:
        """
        Check if the specified axis is in motion.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to check
            
        Returns:
            bool: True if the axis is in motion, False otherwise
        """
        self.select_axis(axis)
        response = self._send_command(":MOTION?")
        return response == "1"
    
    def is_ready(self, axis: Union[AxisName, int, str]) -> bool:
        """
        Check if the specified axis is ready.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to check
            
        Returns:
            bool: True if the axis is ready, False otherwise
        """
        self.select_axis(axis)
        response = self._send_command(":READY?")
        return response == "1"
    
    def is_emergency_stop_active(self) -> bool:
        """
        Check if emergency stop is active.
        
        Returns:
            bool: True if emergency stop is active, False otherwise
        """
        response = self._send_command("EMS?")
        return response == "1"
    
    def get_all_axes_motion_status(self) -> Dict[str, bool]:
        """
        Get motion status of all axes.
        
        Returns:
            Dict[str, bool]: Motion status dictionary
        """
        response = int(self._send_command("MOTIONA?"))
        
        status = {
            "X": bool(response & 0b00000001),
            "Y": bool(response & 0b00000010),
            "Z": bool(response & 0b00000100),
            "U": bool(response & 0b00001000),
            "V": bool(response & 0b00010000),
            "W": bool(response & 0b00100000),
        }
        
        return status
    
    def get_controller_version(self) -> str:
        """
        Get controller version.
        
        Returns:
            str: Controller version
        """
        response = self._send_command("*IDN?")
        return response
    
    # -------------------------------------------------------------------------
    # I/O Commands
    # -------------------------------------------------------------------------
    
    def get_input_status(self, input_number: int) -> bool:
        """
        Get status of the specified input.
        
        Args:
            input_number (int): Input number (0-47)
            
        Returns:
            bool: Input status (True if ON, False if OFF)
        """
        if not 0 <= input_number <= 47:
            raise ValueError("Input number must be between 0 and 47")
        
        response = self._send_command(f"IN{input_number:02d}?")
        return response == "1"
    
    def get_input_port_status(self, port_number: int) -> int:
        """
        Get status of the specified input port (16 bits).
        
        Args:
            port_number (int): Port number (0-2)
            
        Returns:
            int: Input port status (0-65535)
        """
        if not 0 <= port_number <= 2:
            raise ValueError("Port number must be between 0 and 2")
        
        response = self._send_command(f"INP{port_number}?")
        return int(response)
    
    def set_output(self, output_number: int, state: bool) -> None:
        """
        Set the specified output.
        
        Args:
            output_number (int): Output number (0-35)
            state (bool): Output state (True for ON, False for OFF)
        """
        if not 0 <= output_number <= 35:
            raise ValueError("Output number must be between 0 and 35")
        
        self._send_command(f"OUT{output_number:02d}_{1 if state else 0}")
    
    def set_output_port(self, port_number: int, value: int) -> None:
        """
        Set the specified output port (12 bits).
        
        Args:
            port_number (int): Port number (0-2)
            value (int): Output port value (0-4095)
        """
        if not 0 <= port_number <= 2:
            raise ValueError("Port number must be between 0 and 2")
        if not 0 <= value <= 4095:
            raise ValueError("Output port value must be between 0 and 4095")
        
        self._send_command(f"OUTP{port_number}_{value}")
    
    def get_output_port_status(self, port_number: int) -> int:
        """
        Get status of the specified output port (12 bits).
        
        Args:
            port_number (int): Port number (0-2)
            
        Returns:
            int: Output port status (0-4095)
        """
        if not 0 <= port_number <= 2:
            raise ValueError("Port number must be between 0 and 2")
        
        response = self._send_command(f"OUTP{port_number}?")
        return int(response)
    
    # -------------------------------------------------------------------------
    # Program Driving Commands
    # -------------------------------------------------------------------------
    
    def select_program(self, program_number: int) -> None:
        """
        Select program number.
        
        Args:
            program_number (int): Program number (0-7)
        """
        if not 0 <= program_number <= 7:
            raise ValueError("Program number must be between 0 and 7")
        
        self._send_command(f"SELPRG {program_number}")
    
    def start_program(self, mode: str = "RUN") -> None:
        """
        Start the selected program.
        
        Args:
            mode (str, optional): Program mode. Defaults to "RUN".
                "RUN": Run the program
                "STEP": Step through the program
        """
        if mode.upper() not in ["RUN", "STEP"]:
            raise ValueError('Mode must be "RUN" or "STEP"')
        
        self._send_command(f"PRG {mode.upper()}")
    
    def get_program_number(self) -> int:
        """
        Get the selected program number.
        
        Returns:
            int: Program number (0-7)
        """
        response = self._send_command("SELPRG?")
        return int(response)
    
    def get_program_status(self) -> str:
        """
        Get the program status.
        
        Returns:
            str: Program status
                "RUN": Program is running
                "STEP": Program is in step mode
                "STOP": Program is stopped
        """
        response = self._send_command("PRG?")
        status_map = {"0": "RUN", "1": "STEP", "2": "STOP"}
        return status_map.get(response, "UNKNOWN")
    
    def delete_program(self, program_number: int) -> None:
        """
        Delete the specified program.
        
        Args:
            program_number (int): Program number (0-7)
        """
        if not 0 <= program_number <= 7:
            raise ValueError("Program number must be between 0 and 7")
        
        self._send_command(f"DELPRG {program_number}")
        # Wait for the delete operation to complete
        time.sleep(0.5)
    
    def set_program_step(self, program_number: int, step_number: int, command: str) -> None:
        """
        Set a program step.
        
        Args:
            program_number (int): Program number (0-7)
            step_number (int): Step number (0-99)
            command (str): Command to set
        """
        if not 0 <= program_number <= 7:
            raise ValueError("Program number must be between 0 and 7")
        if not 0 <= step_number <= 99:
            raise ValueError("Step number must be between 0 and 99")
        
        self._send_command(f"SETPRG {program_number}, {step_number}, {command}")
        # Wait for the set operation to complete
        time.sleep(0.03)
    
    def get_program_step(self, program_number: int, step_number: int) -> str:
        """
        Get a program step.
        
        Args:
            program_number (int): Program number (0-7)
            step_number (int): Step number (0-99)
            
        Returns:
            str: Program step command
        """
        if not 0 <= program_number <= 7:
            raise ValueError("Program number must be between 0 and 7")
        if not 0 <= step_number <= 99:
            raise ValueError("Step number must be between 0 and 99")
        
        response = self._send_command(f"GETPRG {program_number}, {step_number}")
        return response
    
    # -------------------------------------------------------------------------
    # High-level utility methods
    # -------------------------------------------------------------------------
    
    def wait_for_stop(self, axis: Union[AxisName, int, str], timeout: float = 30.0) -> bool:
        """
        Wait for the specified axis to stop.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to wait for
            timeout (float, optional): Timeout in seconds. Defaults to 30.0.
            
        Returns:
            bool: True if the axis stopped within the timeout, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.is_in_motion(axis):
                return True
            time.sleep(0.1)
        
        return False
    
    def wait_for_all_axes_stop(self, timeout: float = 30.0) -> bool:
        """
        Wait for all axes to stop.
        
        Args:
            timeout (float, optional): Timeout in seconds. Defaults to 30.0.
            
        Returns:
            bool: True if all axes stopped within the timeout, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_all_axes_motion_status()
            if not any(status.values()):
                return True
            time.sleep(0.1)
        
        return False
    
    def home_all_axes(self, timeout: float = 60.0) -> bool:
        """
        Home all axes.
        
        Args:
            timeout (float, optional): Timeout in seconds. Defaults to 60.0.
            
        Returns:
            bool: True if all axes were homed successfully, False otherwise
        """
        # Home X axis
        # Start homing both axes simultaneously
        self.drive(AxisName.X, DriveMode.HOME)
        self.drive(AxisName.Y, DriveMode.HOME)

        # Wait for both axes to complete homing
        while not self.wait_for_stop(AxisName.X, timeout=timeout) or not self.wait_for_stop(AxisName.Y, timeout=timeout):
            pass

        # Return True only if both succeeded
        return True
    
    def move_to_position(self, positions: Dict[Union[AxisName, int, str], float], timeout: float = 30.0) -> bool:
        """
        Move to the specified positions.
        
        Args:
            positions (Dict[Union[AxisName, int, str], float]): Dictionary of axis positions
                Key: Axis (X, Y, Z, U, V, W)
                Value: Absolute position
            timeout (float, optional): Timeout in seconds. Defaults to 30.0.
            
        Returns:
            bool: True if all axes reached their positions within the timeout, False otherwise
        """
        # Set absolute positions
        for axis, position in positions.items():
            self.drive_absolute(axis, position)
        
        # Wait for all axes to stop
        return self.wait_for_all_axes_stop(timeout=timeout)
    
    def jog(self, axis: Union[AxisName, int, str], direction: Union[Direction, int, str]) -> None:
        """
        Jog the specified axis in the specified direction.
        
        Args:
            axis (Union[AxisName, int, str]): Axis to jog
            direction (Union[Direction, int, str]): Jog direction
                0 or CW: Jog in CW direction
                1 or CCW: Jog in CCW direction
        """
        if isinstance(direction, Direction):
            if direction == Direction.CW:
                self.drive(axis, DriveMode.CWJ)
            else:
                self.drive(axis, DriveMode.CCWJ)
        elif isinstance(direction, int) and direction in [0, 1]:
            if direction == 0:
                self.drive(axis, DriveMode.CWJ)
            else:
                self.drive(axis, DriveMode.CCWJ)
        elif isinstance(direction, str):
            if direction.upper() == "CW":
                self.drive(axis, DriveMode.CWJ)
            elif direction.upper() == "CCW":
                self.drive(axis, DriveMode.CCWJ)
            else:
                raise ValueError("Invalid direction. Must be 'CW' or 'CCW'")
        else:
            raise ValueError("Invalid direction. Must be Direction.CW, Direction.CCW, 0, 1, 'CW', or 'CCW'")
