Truncating a fallible callback message to fit ``message_out`` no longer cuts a ``\xhh`` escape or a doubled backslash in half.
The cut falls before the first escape that does not fit, so the written message always ends on an escape boundary and can stop up to three bytes short of the buffer.
