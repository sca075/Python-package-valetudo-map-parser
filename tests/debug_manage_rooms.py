#!/usr/bin/env python3
"""
Debug script for manage_rooms.py
"""

import asyncio
import sys
import traceback

from tests.manage_rooms import main


try:
    asyncio.run(main())
except Exception:
    traceback.print_exc(file=sys.stdout)
