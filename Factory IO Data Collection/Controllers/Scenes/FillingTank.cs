//-----------------------------------------------------------------------------
// FACTORY I/O (SDK)
//
// Copyright (C) Real Games. All rights reserved.
//-----------------------------------------------------------------------------

using System;
using EngineIO;

namespace Controllers
{
    public class FillingTank : Controller
    {
        MemoryFloat fillValve = MemoryMap.Instance.GetFloat("Tank 1 (Fill Valve)", MemoryType.Output);
        MemoryFloat dischargeValve = MemoryMap.Instance.GetFloat("Tank 1 (Discharge Valve)", MemoryType.Output);

        MemoryFloat levelMeter = MemoryMap.Instance.GetFloat("Tank 1 (Level Meter)", MemoryType.Input);

        MemoryBit fillButton = MemoryMap.Instance.GetBit("Fill", MemoryType.Input);
        MemoryBit dischargeButton = MemoryMap.Instance.GetBit("Discharge", MemoryType.Input);
        MemoryBit fillingLight = MemoryMap.Instance.GetBit("Filling", MemoryType.Output);
        MemoryBit dischargingLight = MemoryMap.Instance.GetBit("Discharging", MemoryType.Output);

        MemoryInt timer = MemoryMap.Instance.GetInt("Timer", MemoryType.Output);

        TOF tofFill = new TOF();
        TOF tofDisch = new TOF();

        public FillingTank()
        {
            tofFill.PT = 8000;
            tofDisch.PT = 8000;

            fillValve.Value = 0.0f;
            dischargeValve.Value = 0.0f;

            fillingLight.Value = false;
            dischargingLight.Value = false;
        }

        public override (float x, float y, float z) Execute(int elapsedMilliseconds, int totalTime, bool full)
        {
            if (levelMeter.Value < 7.0f && !full)
            {
                fillValve.Value = 8.4f;
                dischargeValve.Value = 0.0f;
            }
            else if (full)
            {
                fillValve.Value = 8.4f;
                dischargeValve.Value = 5.0f;
            }

            return (x: levelMeter.Value, y: levelMeter.Value, z: levelMeter.Value);
        }
    }
}
