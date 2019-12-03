//-----------------------------------------------------------------------------
// FACTORY I/O (SDK)
//
// Copyright (C) Real Games. All rights reserved.
//-----------------------------------------------------------------------------

namespace Controllers
{
    public abstract class Controller
    {
        public abstract (float x, float y, float z) Execute(int elapsedMilliseconds, int totalTime, bool full);
    }
}
