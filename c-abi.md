# C ABI (proposal)

## C datatypes
| C Type      | Description              |     Size (B) |
|:------------|:-------------------------|-------------:|
| char        | Character / byte         |            8 |
| short       | Short integer            |           16 |
| int         | Integer                  |           32 |
| long        | Long integer             |           64 |
| long long   | Long long integer        |           64 |
| T*          | Pointer                  |           64 |
| float       | Single-precision float   |           32 |
| double      | Double-precision float   |           64 |
| long double | Extended-precision float | **Bikeshed** |

## Registers
| Register | ABI Name | Description    | Saver  |
|:---------|:---------|:---------------|:-------|
| `r0`     | —        | Zero register  | N/A    |
| `r1`     | `ra`     | Return address | Caller |
| `r2`     | `sp`     | Stack pointer  | Callee |
| `r3`     | `tp`     | Thread pointer | N/A    |

**TODO:** Parameters

**TODO:** Saved

**TODO:** Temp

