//-----------------------------------------------------------------------------
// FACTORY I/O (SDK)
//
// Copyright (C) Real Games. All rights reserved.
//-----------------------------------------------------------------------------

using System;
using System.Threading;
using System.Diagnostics;

using EngineIO;

namespace Controllers
{
    class Program
    {
        /// <summary>
        /// Cycle time in milliseconds.
        /// </summary>
        public const int CycleTime = 50;
        public const int EndTime = 240000;
        public int HeightTime = 0;
        public float height = 0;
        /// <summary>
        /// The idea of this sample is to demonstrate that Microsoft Visual Studio can be used as a soft PLC to 
        /// control FACTORY I/O (requires Ultimate Edition).
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            int numCycles = 0;
            //Stopwatch used to measure elapsed time between cycles
            Stopwatch stopwatch = new Stopwatch();
            Stopwatch totalTime = new Stopwatch();

            //MemoryBit used to switch FACTORY I/O between edit and run mode
            MemoryBit start = MemoryMap.Instance.GetBit(MemoryMap.BitCount - 16, MemoryType.Output);

            //MemoryBit used to detect if FACTORY I/O is edit or run mode
            MemoryBit running = MemoryMap.Instance.GetBit(MemoryMap.BitCount - 16, MemoryType.Input);

            //Forcing a rising edge on the start MemoryBit so FACTORY I/O can detect it
            SwitchToRun(start);

            //Controller controller = new FillingTank();
            Controller controller = new PickPlaceXYZ();

            Console.WriteLine(string.Format("Running controller: {0}", controller.GetType().Name));
            Console.WriteLine("Press Escape to shutdown...");

            stopwatch.Start();

            Thread.Sleep(CycleTime);

            totalTime.Start();

            //float tankHeight;
            bool full = false;
            using (System.IO.StreamWriter timeFile = new System.IO.StreamWriter(@"C:\Users\96kai\OneDrive\Documents\GitHub\Cyber-Sec\Data\PickAndPlace\CompromisedScenario\Time10.txt"))
            {
                using (System.IO.StreamWriter xPosFile = new System.IO.StreamWriter(@"C:\Users\96kai\OneDrive\Documents\GitHub\Cyber-Sec\Data\PickAndPlace\CompromisedScenario\XPosition10.txt"))
                {
                    using (System.IO.StreamWriter yPosFile = new System.IO.StreamWriter(@"C:\Users\96kai\OneDrive\Documents\GitHub\Cyber-Sec\Data\PickAndPlace\CompromisedScenario\YPosition10.txt"))
                    {
                        using (System.IO.StreamWriter zPosFile = new System.IO.StreamWriter(@"C:\Users\96kai\OneDrive\Documents\GitHub\Cyber-Sec\Data\PickAndPlace\CompromisedScenario\ZPosition10.txt"))
                        {

                            while (!(Console.KeyAvailable && (Console.ReadKey(false).Key == ConsoleKey.Escape)) && totalTime.ElapsedMilliseconds < EndTime)
                            {
                                //Update the memory map before executing the controller
                                MemoryMap.Instance.Update();

                                if (running.Value)
                                {
                                    stopwatch.Stop();

                                    //tankHeight = controller.Execute((int)stopwatch.ElapsedMilliseconds, (int)totalTime.ElapsedMilliseconds, full);
                                    var position = controller.Execute((int)stopwatch.ElapsedMilliseconds, (int)totalTime.ElapsedMilliseconds, full);
                                    //if (tankHeight < 7.0f && !full)
                                    //{
                                    //    full = false;
                                    //}
                                    //else
                                    //{
                                    //    full = true;
                                    //}

                                    //heightFile.WriteLine(tankHeight);

                                    //timeFile.WriteLine(totalTime.ElapsedMilliseconds);

                                    if((numCycles % 4) == 0)
                                    {
                                        xPosFile.WriteLine(position.x);
                                        yPosFile.WriteLine(position.y);
                                        zPosFile.WriteLine(position.z);

                                        timeFile.WriteLine(totalTime.ElapsedMilliseconds);
                                    }
                                    

                                    stopwatch.Restart();
                                }
                                numCycles ++;
                                Thread.Sleep(CycleTime);
                            }
                        }
                    }
                }
            }

            Shutdown(start);
        }

        static void SwitchToRun(MemoryBit start)
        {
            start.Value = false;
            MemoryMap.Instance.Update();
            Thread.Sleep(500);

            start.Value = true;
            MemoryMap.Instance.Update();
            Thread.Sleep(500);
        }

        static void Shutdown(MemoryBit start)
        {
            start.Value = false;

            MemoryMap.Instance.Update();
            MemoryMap.Instance.Dispose();
        }
    }
}
