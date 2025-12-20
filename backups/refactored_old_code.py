# Hypfer Image Handler Class Rooms Search original
# room_properties = {}
# self.rooms_pos = []
# pixel_size = json_data.get("pixelSize", [])
#
# for layer in json_data.get("layers", []):
#     if layer["__class"] == "MapLayer":
#         meta_data = layer.get("metaData", {})
#         segment_id = meta_data.get("segmentId")
#         if segment_id is not None:
#             name = meta_data.get("name")
#             compressed_pixels = layer.get("compressedPixels", [])
#             pixels = self.data.sublist(compressed_pixels, 3)
#             # Calculate x and y min/max from compressed pixels
#             (
#                 x_min,
#                 y_min,
#                 x_max,
#                 y_max,
#             ) = await self.data.async_get_rooms_coordinates(pixels, pixel_size)
#             corners = self.get_corners(x_max, x_min, y_max, y_min)
#             room_id = str(segment_id)
#             self.rooms_pos.append(
#                 {
#                     "name": name,
#                     "corners": corners,
#                 }
#             )
#             room_properties[room_id] = {
#                 "number": segment_id,
#                 "outline": corners,
#                 "name": name,
#                 "x": ((x_min + x_max) // 2),
#                 "y": ((y_min + y_max) // 2),
#             }
# if room_properties:
#     rooms = RoomStore(self.file_name, room_properties)
#     LOGGER.debug(
#         "%s: Rooms data extracted! %s", self.file_name, rooms.get_rooms()
#     )
# else:
#     LOGGER.debug("%s: Rooms data not available!", self.file_name)
#     self.rooms_pos = None
# return room_properties
